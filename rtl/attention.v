// Multi-head self-attention for single-token autoregressive inference (W8A16)
//
// Optimisations vs original:
//   - qkv_buf [384x16 = 6144-bit register] replaced by M10K (384x16 SPRAM)
//   - head_out_buf [128x16 = 2048-bit register] replaced by M10K (128x16 SPRAM)
//   - out_vec_o [128x16 = 2048-bit register] replaced by streaming output
//     (out_valid_o / out_data_o / out_addr_o) — caller captures into M10K
//   - Both matvec instances use new streaming interface
//
// Flow: QKV matvec -> KV store -> 8x(score, softmax, AV) -> proj matvec
// Streaming output: proj result emitted element-by-element via out_valid_o

module attention (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  input  wire [2047:0] x_i,

  output wire [5:0]    w_sel_o,
  output wire [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,

  output reg           k_we_o,
  output reg  [15:0]   k_wdata_o,
  input  wire [15:0]   k_rdata_i,

  output reg           v_we_o,
  output reg  [15:0]   v_wdata_o,
  input  wire [15:0]   v_rdata_i,

  output reg  [1:0]    kv_layer_o,
  output reg  [2:0]    kv_head_o,
  output reg  [7:0]    kv_pos_o,
  output reg  [3:0]    kv_dim_o,

  // Streaming output (replaces flat out_vec_o [2047:0])
  output wire          out_valid_o,
  output wire [15:0]   out_data_o,
  output wire [6:0]    out_addr_o,

  output reg           done_o
);

  `include "weight_scales.vh"

  localparam [15:0] INV_SQRT_DK = 16'h3400;

  localparam [3:0] S_IDLE      = 4'd0,
                   S_QKV       = 4'd1,
                   S_KV_STORE  = 4'd2,
                   S_SCORE     = 4'd3,
                   S_SCORE_PAD = 4'd4,
                   S_SM_WAIT   = 4'd5,
                   S_AV        = 4'd6,
                   S_AV_STORE  = 4'd7,
                   S_NEXT_HEAD = 4'd8,
                   S_PROJ      = 4'd9,
                   S_DONE      = 4'd10;

  reg [3:0] state;
  reg [1:0] layer_r;
  reg [7:0] pos_r;

  reg [15:0] qkv_scale;
  reg [15:0] proj_scale;
  always @(*) begin
    case (layer_r)
      2'd0: begin qkv_scale = SCALE_BLOCK0_ATTN_QKV_WEIGHT; proj_scale = SCALE_BLOCK0_ATTN_PROJ_WEIGHT; end
      2'd1: begin qkv_scale = SCALE_BLOCK1_ATTN_QKV_WEIGHT; proj_scale = SCALE_BLOCK1_ATTN_PROJ_WEIGHT; end
      2'd2: begin qkv_scale = SCALE_BLOCK2_ATTN_QKV_WEIGHT; proj_scale = SCALE_BLOCK2_ATTN_PROJ_WEIGHT; end
      default: begin qkv_scale = SCALE_BLOCK3_ATTN_QKV_WEIGHT; proj_scale = SCALE_BLOCK3_ATTN_PROJ_WEIGHT; end
    endcase
  end

  reg [2:0] head_idx;

  // -----------------------------------------------------------------------
  // qkv_ram: M10K 384x16 — replaces qkv_buf register
  // Written by u_qkv streaming output, read during KV_STORE and SCORE
  // -----------------------------------------------------------------------
  (* ramstyle = "M10K" *) reg [15:0] qkv_ram [0:383];
  reg  [8:0]  qkv_ram_waddr;
  reg  [15:0] qkv_ram_wdata;
  reg         qkv_ram_we;
  reg  [8:0]  qkv_ram_raddr;
  reg  [15:0] qkv_ram_rdata_r;
  wire [15:0] qkv_ram_rdata = qkv_ram_rdata_r;

  always @(posedge clk_i) begin
    if (qkv_ram_we)
      qkv_ram[qkv_ram_waddr] <= qkv_ram_wdata;
    qkv_ram_rdata_r <= qkv_ram[qkv_ram_raddr];
  end

  // -----------------------------------------------------------------------
  // head_out_ram: M10K 128x16 — replaces head_out_buf register
  // Written during AV_STORE, read by u_proj
  // -----------------------------------------------------------------------
  (* ramstyle = "M10K" *) reg [15:0] head_out_ram [0:127];
  reg  [6:0]  hor_waddr;
  reg  [15:0] hor_wdata;
  reg         hor_we;
  reg  [6:0]  hor_raddr;
  reg  [15:0] hor_rdata_r;

  always @(posedge clk_i) begin
    if (hor_we)
      head_out_ram[hor_waddr] <= hor_wdata;
    hor_rdata_r <= head_out_ram[hor_raddr];
  end

  // head_out_ram read bus for proj matvec: we need full IN_DIM*16 bus
  // Since matvec reads in_vec_i[col*16 +: 16] each cycle, we stream
  // head_out_ram into a shift register as proj runs.
  // Actually matvec_fp16 needs the FULL in_vec_i bus combinationally each cycle.
  // We must materialise it. Use a 128x16 register filled from head_out_ram
  // over 128 cycles before starting proj. This reg is the minimum needed.
  reg [2047:0] proj_in_buf;
  reg          proj_buf_loading;
  reg [7:0]    proj_buf_idx;

  // -----------------------------------------------------------------------
  // QKV matvec — streaming output captured into qkv_ram
  // -----------------------------------------------------------------------
  reg         qkv_start;
  wire [15:0] qkv_w_addr;
  wire        qkv_out_valid;
  wire [15:0] qkv_out_data;
  wire [8:0]  qkv_out_addr;   // 0..383
  wire        qkv_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(384)) u_qkv (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (qkv_start),
    .in_vec_i     (x_i),
    .scale_i      (qkv_scale),
    .weight_addr_o(qkv_w_addr),
    .weight_data_i(w_data_i),
    .out_valid_o  (qkv_out_valid),
    .out_data_o   (qkv_out_data),
    .out_addr_o   (qkv_out_addr),
    .done_o       (qkv_done)
  );

  // Capture qkv streaming output into qkv_ram
  always @(*) begin
    qkv_ram_we    = qkv_out_valid;
    qkv_ram_waddr = qkv_out_addr;
    qkv_ram_wdata = qkv_out_data;
  end

  // -----------------------------------------------------------------------
  // Proj matvec — streaming output passed directly to caller
  // -----------------------------------------------------------------------
  reg          proj_start;
  wire [13:0]  proj_w_addr;
  wire         proj_out_valid;
  wire [15:0]  proj_out_data;
  wire [6:0]   proj_out_addr;
  wire         proj_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(128)) u_proj (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (proj_start),
    .in_vec_i     (proj_in_buf),
    .scale_i      (proj_scale),
    .weight_addr_o(proj_w_addr),
    .weight_data_i(w_data_i),
    .out_valid_o  (proj_out_valid),
    .out_data_o   (proj_out_data),
    .out_addr_o   (proj_out_addr),
    .done_o       (proj_done)
  );

  // Pass proj streaming output straight to module output
  assign out_valid_o = proj_out_valid;
  assign out_data_o  = proj_out_data;
  assign out_addr_o  = proj_out_addr;

  // -----------------------------------------------------------------------
  // Softmax
  // -----------------------------------------------------------------------
  reg         sm_start;
  reg         sm_in_valid;
  reg  [23:0] sm_in_data;
  wire        sm_out_valid;
  wire [15:0] sm_out_data;
  wire        sm_done;

  softmax #(.N(256), .IN_W(24)) u_sm (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (sm_start),
    .in_valid_i (sm_in_valid),
    .in_data_i  (sm_in_data),
    .in_ready_o (),
    .out_valid_o(sm_out_valid),
    .out_data_o (sm_out_data),
    .done_o     (sm_done)
  );

  // attn_buf: 256x16 softmax outputs (Q1.15) — kept as register array
  // (256x16=4096 bits, borderline for M10K; keep as-is for now)
  reg [15:0] attn_buf [0:255];

  // AV accumulators: 16 x fp16
  reg [15:0] av_acc [0:15];

  reg [8:0]  kv_cnt;
  reg [4:0]  sc_cnt;
  reg [7:0]  sc_pos;
  reg [15:0] score_acc;
  reg [1:0]  sc_valid;
  reg [3:0]  sc_dim_d1, sc_dim_d2;
  reg [4:0]  av_cnt;
  reg [7:0]  av_pos;
  reg [1:0]  av_valid;
  reg [3:0]  av_dim_d1, av_dim_d2;
  reg [8:0]  sm_out_cnt;
  reg [8:0]  pad_cnt;

  // KV store: read from qkv_ram (2-cycle latency)
  // K: indices 128..255, V: indices 256..383
  // kv_cnt[6:0] = dim index within K or V
  wire [6:0] kv_idx = kv_cnt[6:0];
  // Issue qkv_ram read 2 cycles ahead — we'll use registered rdata
  // During KV_STORE we read qkv_ram sequentially: K first then V
  reg [15:0] kv_rdata_d1, kv_rdata_d2;  // 2-cycle pipeline from qkv_ram

  // Q extraction from qkv_ram: head_idx*16 + sc_dim
  // We read qkv_ram 2 cycles ahead of needing it in score pipeline
  // sc_dim_d2 is the dimension we're computing for in MAC

  // Score pipeline
  wire [15:0] sc_mac_prod;
  fp16_mul_comb u_sc_mul (
    .a_i(qkv_ram_rdata),   // Q[sc_dim] read from qkv_ram (2-cycle latency)
    .b_i(k_rdata_i),
    .prod_o(sc_mac_prod)
  );

  wire [15:0] sc_mac_sum;
  fp16_add_comb u_sc_add (
    .a_i(score_acc),
    .b_i(sc_mac_prod),
    .sum_o(sc_mac_sum)
  );

  wire [15:0] sc_scaled;
  fp16_mul_comb u_sc_scale (
    .a_i(sc_mac_sum),
    .b_i(INV_SQRT_DK),
    .prod_o(sc_scaled)
  );

  wire [23:0] sc_q167;
  fp16_to_q167 u_sc_cvt (
    .val_i(sc_scaled),
    .q167_o(sc_q167)
  );

  // AV pipeline
  wire [15:0] av_attn_fp16;
  q115_to_fp16 u_av_cvt (
    .val_i(attn_buf[av_pos]),
    .fp16_o(av_attn_fp16)
  );

  wire [15:0] av_mac_prod;
  fp16_mul_comb u_av_mul (
    .a_i(av_attn_fp16),
    .b_i(v_rdata_i),
    .prod_o(av_mac_prod)
  );

  wire [15:0] av_mac_sum;
  fp16_add_comb u_av_add (
    .a_i(av_acc[av_dim_d2]),
    .b_i(av_mac_prod),
    .sum_o(av_mac_sum)
  );

  // Weight store mux
  reg [5:0]  w_sel_r;
  reg [15:0] w_addr_r;
  assign w_sel_o  = w_sel_r;
  assign w_addr_o = w_addr_r;

  always @(*) begin
    case (state)
      S_QKV:  begin w_sel_r = {layer_r, 3'b000} + 6'd4; w_addr_r = qkv_w_addr; end
      S_PROJ: begin w_sel_r = {layer_r, 3'b000} + 6'd5; w_addr_r = {2'b00, proj_w_addr}; end
      default: begin w_sel_r = 6'd0; w_addr_r = 16'd0; end
    endcase
  end

  integer j;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state            <= S_IDLE;
      done_o           <= 1'b0;
      qkv_start        <= 1'b0;
      proj_start       <= 1'b0;
      sm_start         <= 1'b0;
      sm_in_valid      <= 1'b0;
      k_we_o           <= 1'b0;
      v_we_o           <= 1'b0;
      hor_we           <= 1'b0;
      proj_buf_loading <= 1'b0;

    end else begin
      done_o      <= 1'b0;
      qkv_start   <= 1'b0;
      proj_start  <= 1'b0;
      sm_start    <= 1'b0;
      sm_in_valid <= 1'b0;
      k_we_o      <= 1'b0;
      v_we_o      <= 1'b0;
      hor_we      <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state     <= S_QKV;
            layer_r   <= layer_i;
            pos_r     <= pos_i;
            qkv_start <= 1'b1;
          end
        end

        // QKV matvec: streaming output captured into qkv_ram by always @(*)
        S_QKV: begin
          if (qkv_done) begin
            state  <= S_KV_STORE;
            kv_cnt <= 9'd0;
            // Pre-issue first qkv_ram read for K[0] = index 128
            qkv_ram_raddr <= 9'd128;
          end
        end

        // Write K[pos] (qkv[128..255]) and V[pos] (qkv[256..383]) to caches
        // qkv_ram has 2-cycle read latency — pre-issue addresses 2 ahead
        // kv_cnt=0: raddr=128 (already issued in S_QKV exit)
        // kv_cnt=1: raddr=129, issue next
        // kv_cnt=2: rdata=qkv[128]=K[0] valid — write to K cache
        S_KV_STORE: begin
          kv_layer_o <= layer_r;
          kv_pos_o   <= pos_r;
          kv_head_o  <= kv_idx[6:4];
          kv_dim_o   <= kv_idx[3:0];

          // Advance read address (2 cycles ahead)
          if (kv_cnt < 9'd254)
            qkv_ram_raddr <= qkv_ram_raddr + 9'd1;

          // Data valid 2 cycles after address issue
          if (kv_cnt >= 9'd2) begin
            if (kv_cnt < 9'd130) begin
              // Writing K: kv_cnt 2..129 -> K[0..127]
              k_we_o    <= 1'b1;
              k_wdata_o <= qkv_ram_rdata;
            end else begin
              // Writing V: kv_cnt 130..257 -> V[0..127]
              v_we_o    <= 1'b1;
              v_wdata_o <= qkv_ram_rdata;
            end
          end

          if (kv_cnt == 9'd257) begin
            state    <= S_SCORE;
            head_idx <= 3'd0;
            sm_start <= 1'b1;
            sc_pos   <= 8'd0;
            sc_cnt   <= 5'd0;
            score_acc <= 16'd0;
            sc_valid <= 2'b00;
            // Pre-issue Q[head=0, dim=0] read: index = 0*16+0 = 0
            qkv_ram_raddr <= 9'd0;
          end
          kv_cnt <= kv_cnt + 9'd1;
        end

        // Score: Q[head][dim] . K[pos][dim] for pos=0..pos_r, dim=0..15
        // Q read from qkv_ram (2-cycle latency), K read from KV cache (2-cycle)
        // We issue qkv_ram read 2 cycles before we need it — same timing as K
        S_SCORE: begin
          kv_layer_o <= layer_r;
          kv_head_o  <= head_idx;
          k_we_o     <= 1'b0;

          sc_dim_d1 <= sc_cnt[3:0];
          sc_dim_d2 <= sc_dim_d1;
          sc_valid  <= {sc_valid[0], (sc_cnt < 5'd16) ? 1'b1 : 1'b0};

          if (sc_cnt < 5'd16) begin
            kv_pos_o      <= sc_pos;
            kv_dim_o      <= sc_cnt[3:0];
            // Issue Q read: qkv_ram[head_idx*16 + sc_cnt]
            qkv_ram_raddr <= {2'b00, head_idx, sc_cnt[3:0]};
          end

          sc_cnt <= sc_cnt + 5'd1;

          if (sc_valid[1]) begin
            if (sc_dim_d2 == 4'd15) begin
              sm_in_valid <= 1'b1;
              sm_in_data  <= sc_q167;
              score_acc   <= 16'd0;
              if (sc_pos == pos_r) begin
                if (pos_r == 8'd255) begin
                  state      <= S_SM_WAIT;
                  sm_out_cnt <= 9'd0;
                end else begin
                  state   <= S_SCORE_PAD;
                  pad_cnt <= {1'b0, pos_r} + 9'd1;
                end
              end else begin
                sc_pos   <= sc_pos + 8'd1;
                sc_cnt   <= 5'd0;
                sc_valid <= 2'b00;
              end
            end else begin
              score_acc <= sc_mac_sum;
            end
          end
        end

        S_SCORE_PAD: begin
          sm_in_valid <= 1'b1;
          sm_in_data  <= 24'sh800000;
          pad_cnt     <= pad_cnt + 9'd1;
          if (pad_cnt == 9'd255) begin
            state      <= S_SM_WAIT;
            sm_out_cnt <= 9'd0;
          end
        end

        S_SM_WAIT: begin
          if (sm_out_valid) begin
            attn_buf[sm_out_cnt[7:0]] <= sm_out_data;
            sm_out_cnt <= sm_out_cnt + 9'd1;
          end
          if (sm_done) begin
            state    <= S_AV;
            av_pos   <= 8'd0;
            av_cnt   <= 5'd0;
            av_valid <= 2'b00;
            for (j = 0; j < 16; j = j + 1)
              av_acc[j] <= 16'd0;
          end
        end

        S_AV: begin
          kv_layer_o <= layer_r;
          kv_head_o  <= head_idx;
          v_we_o     <= 1'b0;

          av_dim_d1 <= av_cnt[3:0];
          av_dim_d2 <= av_dim_d1;
          av_valid  <= {av_valid[0], (av_cnt < 5'd16) ? 1'b1 : 1'b0};

          if (av_cnt < 5'd16) begin
            kv_pos_o <= av_pos;
            kv_dim_o <= av_cnt[3:0];
          end

          av_cnt <= av_cnt + 5'd1;

          if (av_valid[1]) begin
            av_acc[av_dim_d2] <= av_mac_sum;
            if (av_dim_d2 == 4'd15) begin
              if (av_pos == pos_r)
                state <= S_AV_STORE;
              else begin
                av_pos   <= av_pos + 8'd1;
                av_cnt   <= 5'd0;
                av_valid <= 2'b00;
              end
            end
          end
        end

        // Write 16 AV accumulators into head_out_ram at [head_idx*16 .. +15]
        // 16 sequential writes over 16 cycles
        S_AV_STORE: begin
          hor_we    <= 1'b1;
          hor_waddr <= {head_idx, av_cnt[3:0]};
          hor_wdata <= av_acc[av_cnt[3:0]];
          av_cnt    <= av_cnt + 5'd1;
          if (av_cnt[3:0] == 4'd15) begin
            hor_we <= 1'b0;
            state  <= S_NEXT_HEAD;
          end
        end

        S_NEXT_HEAD: begin
          if (head_idx == 3'd7) begin
            // Load proj_in_buf from head_out_ram before starting proj
            proj_buf_loading <= 1'b1;
            proj_buf_idx     <= 8'd0;
            hor_raddr        <= 7'd0;
            state            <= S_PROJ;
          end else begin
            head_idx  <= head_idx + 3'd1;
            state     <= S_SCORE;
            sm_start  <= 1'b1;
            sc_pos    <= 8'd0;
            sc_cnt    <= 5'd0;
            score_acc <= 16'd0;
            sc_valid  <= 2'b00;
            qkv_ram_raddr <= {2'b00, head_idx + 3'd1, 4'd0};
          end
        end

        // S_PROJ: first 130 cycles load proj_in_buf from head_out_ram,
        // then start proj matvec (proj runs while state stays S_PROJ)
        S_PROJ: begin
          if (proj_buf_loading) begin
            // Advance read address
            if (proj_buf_idx < 8'd127)
              hor_raddr <= hor_raddr + 7'd1;
            // Capture data (2-cycle latency)
            if (proj_buf_idx >= 8'd2)
              proj_in_buf[(proj_buf_idx - 8'd2)*16 +: 16] <= hor_rdata_r;
            proj_buf_idx <= proj_buf_idx + 8'd1;
            if (proj_buf_idx == 8'd129) begin
              proj_buf_loading <= 1'b0;
              proj_start       <= 1'b1;
            end
          end
          if (proj_done) begin
            state <= S_DONE;
          end
        end

        S_DONE: begin
          done_o <= 1'b1;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
