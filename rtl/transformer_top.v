// Transformer top: full inference pipeline (W8A16)
//
// Optimisations:
//   - u_head_proj uses streaming matvec_fp16 (no 4096-bit out_vec_o register)
//     streaming output captured into logits_ram (M10K 256x16)
//   - sampler reads from logits_ram sequentially (no 4096-bit logits_i bus)
//   - transformer_layer.out_vec_o still used as x_reg source (2048-bit bus)
//     but this is unavoidable — it's the hidden state passed between layers

module transformer_top (
  input  wire        clk_i,
  input  wire        rst_i,

  input  wire [7:0]  token_i,
  input  wire        start_i,
  input  wire        generate_i,

  output reg  [5:0]  w_sel_o,
  output reg  [15:0] w_addr_o,
  input  wire [7:0]  w_data_i,

  output wire        k_we_o,
  output wire [15:0] k_wdata_o,
  input  wire [15:0] k_rdata_i,

  output wire        v_we_o,
  output wire [15:0] v_wdata_o,
  input  wire [15:0] v_rdata_i,

  output wire [1:0]  kv_layer_o,
  output wire [2:0]  kv_head_o,
  output wire [7:0]  kv_pos_o,
  output wire [3:0]  kv_dim_o,

  output reg  [7:0]  token_o,
  output reg         token_valid_o,
  output reg         busy_o,
  output reg         done_o
);

  `include "weight_scales.vh"

  localparam [3:0] S_IDLE         = 4'd0,
                   S_EMBED        = 4'd1,
                   S_LAYER_START  = 4'd2,
                   S_LAYER_WAIT   = 4'd3,
                   S_LN_F_START   = 4'd4,
                   S_LN_F_WAIT    = 4'd5,
                   S_HEAD_PROJ    = 4'd6,
                   S_SAMPLE       = 4'd7,
                   S_TOKEN_OUT    = 4'd8;

  reg [3:0] state;

  reg [2047:0] x_reg;
  reg [7:0]    cur_token;
  reg [7:0]    pos_r;
  reg [1:0]    layer_idx;
  reg          generating;

  // -----------------------------------------------------------------------
  // Embedding
  // -----------------------------------------------------------------------
  reg         emb_start;
  wire [5:0]  emb_w_sel;
  wire [15:0] emb_w_addr;
  wire [2047:0] emb_out;
  wire        emb_done;

  embedding u_emb (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(emb_start),
    .token_id_i(cur_token), .position_i(pos_r),
    .tok_scale_i(SCALE_TOK_EMB_WEIGHT), .pos_scale_i(SCALE_POS_EMB_WEIGHT),
    .w_sel_o(emb_w_sel), .w_addr_o(emb_w_addr), .w_data_i(w_data_i),
    .embed_o(emb_out), .done_o(emb_done), .busy_o()
  );

  // -----------------------------------------------------------------------
  // Transformer layer (single instance, reused 4 times)
  // -----------------------------------------------------------------------
  reg          tl_start;
  wire [5:0]   tl_w_sel;
  wire [15:0]  tl_w_addr;
  wire [2047:0] tl_out;
  wire         tl_done;
  wire         tl_k_we, tl_v_we;
  wire [15:0]  tl_k_wdata, tl_v_wdata;
  wire [1:0]   tl_kv_layer;
  wire [2:0]   tl_kv_head;
  wire [7:0]   tl_kv_pos;
  wire [3:0]   tl_kv_dim;

  transformer_layer u_tl (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(tl_start),
    .layer_i(layer_idx), .pos_i(pos_r), .x_i(x_reg),
    .w_sel_o(tl_w_sel), .w_addr_o(tl_w_addr), .w_data_i(w_data_i),
    .k_we_o(tl_k_we), .k_wdata_o(tl_k_wdata), .k_rdata_i(k_rdata_i),
    .v_we_o(tl_v_we), .v_wdata_o(tl_v_wdata), .v_rdata_i(v_rdata_i),
    .kv_layer_o(tl_kv_layer), .kv_head_o(tl_kv_head),
    .kv_pos_o(tl_kv_pos), .kv_dim_o(tl_kv_dim),
    .out_vec_o(tl_out), .done_o(tl_done)
  );

  reg kv_active;
  assign k_we_o     = kv_active ? tl_k_we    : 1'b0;
  assign k_wdata_o  = kv_active ? tl_k_wdata : 16'd0;
  assign v_we_o     = kv_active ? tl_v_we    : 1'b0;
  assign v_wdata_o  = kv_active ? tl_v_wdata : 16'd0;
  assign kv_layer_o = kv_active ? tl_kv_layer : 2'd0;
  assign kv_head_o  = kv_active ? tl_kv_head  : 3'd0;
  assign kv_pos_o   = kv_active ? tl_kv_pos   : 8'd0;
  assign kv_dim_o   = kv_active ? tl_kv_dim   : 4'd0;

  // -----------------------------------------------------------------------
  // Final LayerNorm
  // -----------------------------------------------------------------------
  reg         lnf_start;
  wire [5:0]  lnf_w_sel;
  wire [6:0]  lnf_w_addr;
  wire        lnf_done;
  wire [2047:0] lnf_y;

  layernorm u_ln_f (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(lnf_start),
    .x_i(x_reg),
    .w_sel_o(lnf_w_sel), .w_addr_o(lnf_w_addr), .w_data_i(w_data_i),
    .gamma_sel_i(6'd34),
    .gamma_scale_i(SCALE_LN_F_WEIGHT), .beta_scale_i(SCALE_LN_F_BIAS),
    .y_o(lnf_y), .done_o(lnf_done), .busy_o()
  );

  // -----------------------------------------------------------------------
  // Head projection: matvec_fp16 128->256, streaming output -> logits_ram
  // -----------------------------------------------------------------------
  reg          head_start;
  wire [14:0]  head_w_addr;
  wire         head_out_valid;
  wire [15:0]  head_out_data;
  wire [7:0]   head_out_addr;  // 0..255
  wire         head_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(256)) u_head_proj (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(head_start),
    .in_vec_i(x_reg), .scale_i(SCALE_TOK_EMB_WEIGHT),
    .weight_addr_o(head_w_addr), .weight_data_i(w_data_i),
    .out_valid_o(head_out_valid), .out_data_o(head_out_data),
    .out_addr_o(head_out_addr),
    .done_o(head_done)
  );

  // logits_ram: M10K 256x16 — replaces 4096-bit logits register
  (* ramstyle = "M10K" *) reg [15:0] logits_ram [0:255];
  always @(posedge clk_i) begin
    if (head_out_valid)
      logits_ram[head_out_addr] <= head_out_data;
  end

  // -----------------------------------------------------------------------
  // Sampler: reads logits_ram sequentially instead of wide flat bus
  // -----------------------------------------------------------------------
  reg         samp_start;
  wire [7:0]  samp_token;
  wire        samp_done;

  // Sampler needs logits one at a time — we stream from logits_ram
  // We use a simple wrapper: after head_done, FSM enters S_SAMPLE,
  // sampler_stream reads logits_ram[0..255] and feeds sampler_seq below.
  // Replace sampler with a version that accepts streaming input.
  // Since sampler.v uses wide bus, we rewrite it inline here as
  // a sequential argmax that reads logits_ram directly.

  reg [1:0]  samp_state;
  reg [8:0]  samp_idx;
  reg [15:0] samp_best_val;
  reg [7:0]  samp_best_idx;
  reg [15:0] samp_rdata_r;
  reg [8:0]  samp_raddr;

  // logits_ram read port for sampler
  always @(posedge clk_i) begin
    samp_rdata_r <= logits_ram[samp_raddr[7:0]];
  end

  wire [15:0] samp_cur = samp_rdata_r;
  wire samp_a_neg  = samp_cur[15];
  wire samp_b_neg  = samp_best_val[15];
  wire samp_mag_gt = samp_cur[14:0] > samp_best_val[14:0];
  wire samp_a_gt_b = (samp_a_neg != samp_b_neg) ? samp_b_neg :
                      samp_a_neg ? ~samp_mag_gt : samp_mag_gt;

  localparam SAMP_IDLE = 2'd0, SAMP_PREFETCH = 2'd1,
             SAMP_SCAN = 2'd2, SAMP_DONE = 2'd3;

  assign samp_token = samp_best_idx;
  assign samp_done  = (samp_state == SAMP_DONE);

  always @(posedge clk_i) begin
    if (rst_i) begin
      samp_state    <= SAMP_IDLE;
      samp_best_val <= 16'hFC00;
      samp_best_idx <= 8'd0;
    end else begin
      case (samp_state)
        SAMP_IDLE: begin
          if (samp_start) begin
            samp_idx      <= 9'd0;
            samp_raddr    <= 9'd0;
            samp_best_val <= 16'hFC00;
            samp_best_idx <= 8'd0;
            samp_state    <= SAMP_PREFETCH;
          end
        end
        // 1-cycle prefetch for BRAM read latency
        SAMP_PREFETCH: begin
          samp_raddr <= 9'd1;
          samp_state <= SAMP_SCAN;
          samp_idx   <= 9'd0;
        end
        SAMP_SCAN: begin
          // samp_cur = logits_ram[samp_idx] (1-cycle BRAM latency already done)
          if (samp_a_gt_b) begin
            samp_best_val <= samp_cur;
            samp_best_idx <= samp_idx[7:0];
          end
          samp_idx   <= samp_idx + 9'd1;
          samp_raddr <= samp_raddr + 9'd1;
          if (samp_idx == 9'd255)
            samp_state <= SAMP_DONE;
        end
        SAMP_DONE: begin
          samp_state <= SAMP_IDLE;
        end
      endcase
    end
  end

  // -----------------------------------------------------------------------
  // Weight store mux
  // -----------------------------------------------------------------------
  always @(*) begin
    case (state)
      S_EMBED: begin
        w_sel_o  = emb_w_sel;
        w_addr_o = emb_w_addr;
      end
      S_LAYER_START, S_LAYER_WAIT: begin
        w_sel_o  = tl_w_sel;
        w_addr_o = tl_w_addr;
      end
      S_LN_F_START, S_LN_F_WAIT: begin
        w_sel_o  = lnf_w_sel;
        w_addr_o = {9'd0, lnf_w_addr};
      end
      S_HEAD_PROJ: begin
        w_sel_o  = 6'd0;
        w_addr_o = {1'b0, head_w_addr};
      end
      default: begin
        w_sel_o  = 6'd0;
        w_addr_o = 16'd0;
      end
    endcase
  end

  always @(*) begin
    kv_active = (state == S_LAYER_START) || (state == S_LAYER_WAIT);
  end

  // -----------------------------------------------------------------------
  // Main FSM
  // -----------------------------------------------------------------------
  always @(posedge clk_i) begin
    if (rst_i) begin
      state         <= S_IDLE;
      done_o        <= 1'b0;
      token_valid_o <= 1'b0;
      busy_o        <= 1'b0;
      emb_start     <= 1'b0;
      tl_start      <= 1'b0;
      lnf_start     <= 1'b0;
      head_start    <= 1'b0;
      samp_start    <= 1'b0;
      generating    <= 1'b0;
      pos_r         <= 8'd0;
      layer_idx     <= 2'd0;

    end else begin
      done_o        <= 1'b0;
      token_valid_o <= 1'b0;
      emb_start     <= 1'b0;
      tl_start      <= 1'b0;
      lnf_start     <= 1'b0;
      head_start    <= 1'b0;
      samp_start    <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            cur_token  <= token_i;
            generating <= generate_i;
            emb_start  <= 1'b1;
            busy_o     <= 1'b1;
            state      <= S_EMBED;
          end
        end

        S_EMBED: begin
          if (emb_done) begin
            x_reg     <= emb_out;
            layer_idx <= 2'd0;
            tl_start  <= 1'b1;
            state     <= S_LAYER_WAIT;
          end
        end

        S_LAYER_START: begin
          tl_start <= 1'b1;
          state    <= S_LAYER_WAIT;
        end

        S_LAYER_WAIT: begin
          if (tl_done) begin
            x_reg     <= tl_out;
            layer_idx <= layer_idx + 2'd1;
            if (layer_idx == 2'd3) begin
              if (!generating) begin
                pos_r  <= pos_r + 8'd1;
                done_o <= 1'b1;
                busy_o <= 1'b0;
                state  <= S_IDLE;
              end else begin
                state <= S_LN_F_START;
              end
            end else begin
              state <= S_LAYER_START;
            end
          end
        end

        S_LN_F_START: begin
          lnf_start <= 1'b1;
          state     <= S_LN_F_WAIT;
        end

        S_LN_F_WAIT: begin
          if (lnf_done) begin
            x_reg      <= lnf_y;
            head_start <= 1'b1;
            state      <= S_HEAD_PROJ;
          end
        end

        S_HEAD_PROJ: begin
          if (head_done) begin
            samp_start <= 1'b1;
            state      <= S_SAMPLE;
          end
        end

        S_SAMPLE: begin
          if (samp_done) begin
            token_o <= samp_token;
            state   <= S_TOKEN_OUT;
          end
        end

        S_TOKEN_OUT: begin
          token_valid_o <= 1'b1;
          pos_r         <= pos_r + 8'd1;
          if (pos_r == 8'd255) begin
            done_o     <= 1'b1;
            busy_o     <= 1'b0;
            generating <= 1'b0;
            state      <= S_IDLE;
          end else begin
            cur_token <= samp_token;
            emb_start <= 1'b1;
            state     <= S_EMBED;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
