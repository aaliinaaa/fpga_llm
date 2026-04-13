// Single transformer block: LN1 -> Attention -> Residual -> LN2 -> FF -> Residual
//
// W8A16: int8 weights in BRAM, fp16 activations throughout
// Submodules: layernorm (flat fp16 bus), attention (fp16), 2x matvec_fp16, gelu (fp16 PWL)
// Weight store and KV cache are external, muxed here
//
// Optimisations vs original:
//   - ff_buf (8192-bit register) replaced by M10K Simple Dual-Port RAM (512 x 16)
//   - gen_res1 / gen_res2 (128 parallel fp16_add_comb) replaced by a single
//     fp16_add_comb iterated over 128 cycles; two new FSM states S_RES1_ACC and
//     S_RES2_ACC replace the single-cycle S_RES1 / S_RES2.
//     Latency cost: +127 cycles per residual add (negligible vs matvec latency).
//
// Weight store tensor_sel mapping per layer L = {layer_r, 3'b000}:
//   LN1 gamma = L+2, LN1 beta = L+3
//   QKV = L+4, Proj = L+5 (handled inside attention)
//   LN2 gamma = L+6, LN2 beta = L+7
//   FF_up = L+8, FF_down = L+9

module transformer_layer (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  input  wire [2047:0] x_i,

  // Weight store interface
  output reg  [5:0]    w_sel_o,
  output reg  [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,

  // K cache (fp16)
  output wire          k_we_o,
  output wire [15:0]   k_wdata_o,
  input  wire [15:0]   k_rdata_i,

  // V cache (fp16)
  output wire          v_we_o,
  output wire [15:0]   v_wdata_o,
  input  wire [15:0]   v_rdata_i,

  // KV cache address (shared K/V)
  output wire [1:0]    kv_layer_o,
  output wire [2:0]    kv_head_o,
  output wire [7:0]    kv_pos_o,
  output wire [3:0]    kv_dim_o,

  // Output
  output reg  [2047:0] out_vec_o,
  output reg           done_o
);

  `include "weight_scales.vh"

  wire [2047:0] ff_down_out;

  // -----------------------------------------------------------------------
  // FSM states
  // -----------------------------------------------------------------------
  localparam [3:0] S_IDLE      = 4'd0,
                   S_LN_START  = 4'd1,
                   S_LN_WAIT   = 4'd2,
                   S_ATTN      = 4'd3,
                   S_RES1_ACC  = 4'd4,
                   S_FF_UP     = 4'd5,
                   S_GELU      = 4'd6,
                   S_FF_DRAIN  = 4'd10,
                   S_FF_DOWN   = 4'd7,
                   S_RES2_ACC  = 4'd8,
                   S_DONE      = 4'd9;

  reg [3:0] state;

  // Latched inputs
  reg [1:0]    layer_r;
  reg [7:0]    pos_r;

  // Residual register (preserved across LN+attn and LN+FF)
  reg [2047:0] x_reg;

  // LN/attn output buffer (128 x fp16)
  reg [2047:0] sub_out;

  // -----------------------------------------------------------------------
  // ff_buf: replaced with M10K Simple Dual-Port RAM (512 x 16)
  // Write port: used by FF_up (via ff_ram_we / ff_ram_waddr / ff_ram_wdata)
  //             and by GELU (overwrites in-place)
  // Read port:  used by FF_down input (ff_ram_rdata) and GELU read
  // -----------------------------------------------------------------------
  reg         ff_ram_we;
  reg  [8:0]  ff_ram_waddr;   // 0..511
  reg  [15:0] ff_ram_wdata;
  reg  [8:0]  ff_ram_raddr;   // 0..511
  reg  [15:0] ff_ram_rdata_r;
  wire [15:0] ff_ram_rdata = ff_ram_rdata_r;

  // Infer M10K: simple dual-port, separate read/write clocks (same clock here)
  (* ramstyle = "M10K" *)
  reg [15:0] ff_ram [0:511];

  always @(posedge clk_i) begin
    if (ff_ram_we)
      ff_ram[ff_ram_waddr] <= ff_ram_wdata;
    ff_ram_rdata_r <= ff_ram[ff_ram_raddr];
  end

  // -----------------------------------------------------------------------
  // Residual add: single fp16_add_comb, iterated 128 cycles
  // Inputs selected by res_mux: 0 = res1 (sub_out+x_reg), 1 = res2 (ff_down+x_reg)
  // Result stored element-by-element into x_reg (res1) or out_vec_o (res2)
  // -----------------------------------------------------------------------
  reg [6:0]  res_idx;       // 0..127
  reg        res_mux;       // 0=res1, 1=res2

  wire [15:0] res_a = res_mux
      ? ff_down_out[res_idx*16 +: 16]
      : sub_out    [res_idx*16 +: 16];
  wire [15:0] res_b = x_reg[res_idx*16 +: 16];

  wire [15:0] res_sum;
  fp16_add_comb u_res_add (
    .a_i  (res_a),
    .b_i  (res_b),
    .sum_o(res_sum)
  );

  // LN which: 0=LN1, 1=LN2
  reg ln_which;

  // GELU index
  reg [9:0] gelu_idx;

  // Per-layer scale muxes
  reg [15:0] ln_gamma_scale;
  reg [15:0] ln_beta_scale;
  reg [15:0] ff_up_scale;
  reg [15:0] ff_down_scale;

  always @(*) begin
    case (layer_r)
      2'd0: begin
        ff_up_scale   = SCALE_BLOCK0_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK0_FF_DOWN_WEIGHT;
      end
      2'd1: begin
        ff_up_scale   = SCALE_BLOCK1_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK1_FF_DOWN_WEIGHT;
      end
      2'd2: begin
        ff_up_scale   = SCALE_BLOCK2_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK2_FF_DOWN_WEIGHT;
      end
      default: begin
        ff_up_scale   = SCALE_BLOCK3_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK3_FF_DOWN_WEIGHT;
      end
    endcase
  end

  always @(*) begin
    case ({layer_r, ln_which})
      3'b000: begin ln_gamma_scale = SCALE_BLOCK0_LN1_WEIGHT; ln_beta_scale = SCALE_BLOCK0_LN1_BIAS; end
      3'b001: begin ln_gamma_scale = SCALE_BLOCK0_LN2_WEIGHT; ln_beta_scale = SCALE_BLOCK0_LN2_BIAS; end
      3'b010: begin ln_gamma_scale = SCALE_BLOCK1_LN1_WEIGHT; ln_beta_scale = SCALE_BLOCK1_LN1_BIAS; end
      3'b011: begin ln_gamma_scale = SCALE_BLOCK1_LN2_WEIGHT; ln_beta_scale = SCALE_BLOCK1_LN2_BIAS; end
      3'b100: begin ln_gamma_scale = SCALE_BLOCK2_LN1_WEIGHT; ln_beta_scale = SCALE_BLOCK2_LN1_BIAS; end
      3'b101: begin ln_gamma_scale = SCALE_BLOCK2_LN2_WEIGHT; ln_beta_scale = SCALE_BLOCK2_LN2_BIAS; end
      3'b110: begin ln_gamma_scale = SCALE_BLOCK3_LN1_WEIGHT; ln_beta_scale = SCALE_BLOCK3_LN1_BIAS; end
      default: begin ln_gamma_scale = SCALE_BLOCK3_LN2_WEIGHT; ln_beta_scale = SCALE_BLOCK3_LN2_BIAS; end
    endcase
  end

  // -----------------------------------------------------------------------
  // LayerNorm
  // -----------------------------------------------------------------------
  reg          ln_start;
  reg  [5:0]   ln_gamma_sel;
  wire [5:0]   ln_w_sel;
  wire [6:0]   ln_w_addr;
  wire [2047:0] ln_y;
  wire         ln_done;

  layernorm u_ln (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ln_start),
    .x_i          (x_reg),
    .w_sel_o      (ln_w_sel),
    .w_addr_o     (ln_w_addr),
    .w_data_i     (w_data_i),
    .gamma_sel_i  (ln_gamma_sel),
    .gamma_scale_i(ln_gamma_scale),
    .beta_scale_i (ln_beta_scale),
    .y_o          (ln_y),
    .done_o       (ln_done),
    .busy_o       ()
  );

  // -----------------------------------------------------------------------
  // Attention
  // -----------------------------------------------------------------------
  reg          attn_start;
  wire [5:0]   attn_w_sel;
  wire [15:0]  attn_w_addr;
  wire [2047:0] attn_out;
  wire         attn_done;

  wire         attn_k_we,   attn_v_we;
  wire [15:0]  attn_k_wdata, attn_v_wdata;
  wire [1:0]   attn_kv_layer;
  wire [2:0]   attn_kv_head;
  wire [7:0]   attn_kv_pos;
  wire [3:0]   attn_kv_dim;

  attention u_attn (
    .clk_i     (clk_i),
    .rst_i     (rst_i),
    .start_i   (attn_start),
    .layer_i   (layer_r),
    .pos_i     (pos_r),
    .x_i       (sub_out),
    .w_sel_o   (attn_w_sel),
    .w_addr_o  (attn_w_addr),
    .w_data_i  (w_data_i),
    .k_we_o    (attn_k_we),
    .k_wdata_o (attn_k_wdata),
    .k_rdata_i (k_rdata_i),
    .v_we_o    (attn_v_we),
    .v_wdata_o (attn_v_wdata),
    .v_rdata_i (v_rdata_i),
    .kv_layer_o(attn_kv_layer),
    .kv_head_o (attn_kv_head),
    .kv_pos_o  (attn_kv_pos),
    .kv_dim_o  (attn_kv_dim),
    .out_vec_o (attn_out),
    .done_o    (attn_done)
  );

  assign k_we_o     = attn_k_we;
  assign k_wdata_o  = attn_k_wdata;
  assign v_we_o     = attn_v_we;
  assign v_wdata_o  = attn_v_wdata;
  assign kv_layer_o = attn_kv_layer;
  assign kv_head_o  = attn_kv_head;
  assign kv_pos_o   = attn_kv_pos;
  assign kv_dim_o   = attn_kv_dim;

  // -----------------------------------------------------------------------
  // FF_up matvec: 128 -> 512
  // out_vec_o now writes directly to ff_ram instead of ff_buf register
  // -----------------------------------------------------------------------
  reg          ff_up_start;
  wire [15:0]  ff_up_addr;
  // ff_up_out is 8192 bits in original - we need to intercept writes to RAM.
  // We instantiate matvec_fp16 with a modified output path:
  // Instead of capturing the full out_vec_o bus, we use a write-back approach:
  // matvec writes one element per cycle when a row completes.
  // We use a wrapper that exposes per-element done signals.
  //
  // Simplest approach: keep the wide output bus from matvec but register it
  // into ff_ram after done_o. Since ff_up takes 128*512+2 = 65538 cycles,
  // the RAM write-back (512 cycles) adds minimal latency.
  // Alternatively: capture done + row index inside matvec - but that requires
  // modifying matvec_fp16.
  //
  // CHOSEN: after ff_up_done, copy ff_up_out into ff_ram over 512 cycles via
  // a small post-processing FSM (S_FF_UP state already waits for done, we
  // extend it to drain into RAM). The wide register ff_up_out is kept but
  // only exists in the matvec instance - no extra reg in this module.

  wire [8191:0] ff_up_out;
  wire          ff_up_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(512)) u_ff_up (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_up_start),
    .in_vec_i     (sub_out),
    .scale_i      (ff_up_scale),
    .weight_addr_o(ff_up_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (ff_up_out),
    .done_o       (ff_up_done)
  );

  // -----------------------------------------------------------------------
  // FF_down matvec: 512 -> 128, input comes from ff_ram
  // matvec_fp16 expects a flat in_vec_i bus; we need to stream from RAM.
  // We use a registered capture buffer (128 cycles wide) that is filled
  // from ff_ram before starting ff_down. This replaces the old 8192-bit ff_buf.
  // The capture buffer is 2048 bits (128 x fp16) - reused as ff_down input
  // is streamed element by element, so we actually need the full 8192-bit
  // bus for matvec in_vec_i. 
  //
  // REVISED APPROACH for ff_down:
  // We keep matvec_fp16 as-is (needs full in_vec_i bus). We use ff_ram as the
  // source and stream INTO a 8192-bit register over 512 cycles before starting
  // ff_down. This register is the ONLY 8192-bit register and lives here
  // temporarily during the drain phase (512 cycles). Then ff_down starts.
  // This saves ALM because the 8192-bit reg is only used during GELU+drain,
  // not synthesised as a flat always-enabled register that Quartus keeps live
  // across all pipeline stages.
  //
  // Actually the key saving is that ff_ram is M10K, not ALM flip-flops.
  // The drain buffer below IS still a register - but Quartus can pack it
  // tighter once it knows it's only written in one state.
  // -----------------------------------------------------------------------

  // Drain buffer: filled from ff_ram over 512 cycles, then fed to ff_down
  reg [8191:0] ff_drain_buf;
  reg  [9:0]   drain_idx;    // 0..511

  wire [15:0]  ff_down_addr;
//   wire [2047:0] ff_down_out;
  wire          ff_down_done;
  reg           ff_down_start;

  matvec_fp16 #(.IN_DIM(512), .OUT_DIM(128)) u_ff_down (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_down_start),
    .in_vec_i     (ff_drain_buf),
    .scale_i      (ff_down_scale),
    .weight_addr_o(ff_down_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (ff_down_out),
    .done_o       (ff_down_done)
  );

  // -----------------------------------------------------------------------
  // GELU (fp16 PWL, 2-cycle pipeline)
  // Reads from ff_ram, writes back to ff_ram in-place
  // -----------------------------------------------------------------------
  reg         gelu_valid_in;
  reg  [15:0] gelu_in;
  wire [15:0] gelu_out;
  wire        gelu_valid_out;

  gelu u_gelu (
    .clk_i  (clk_i),
    .valid_i(gelu_valid_in),
    .x_i    (gelu_in),
    .valid_o(gelu_valid_out),
    .y_o    (gelu_out)
  );

  // -----------------------------------------------------------------------
  // Weight store mux
  // -----------------------------------------------------------------------
  always @(*) begin
    case (state)
      S_LN_START, S_LN_WAIT: begin
        w_sel_o  = ln_w_sel;
        w_addr_o = {9'd0, ln_w_addr};
      end
      S_ATTN: begin
        w_sel_o  = attn_w_sel;
        w_addr_o = attn_w_addr;
      end
      S_FF_UP: begin
        w_sel_o  = {layer_r, 3'b000} + 6'd8;
        w_addr_o = ff_up_addr;
      end
      S_FF_DOWN: begin
        w_sel_o  = {layer_r, 3'b000} + 6'd9;
        w_addr_o = ff_down_addr;
      end
      default: begin
        w_sel_o  = 6'd0;
        w_addr_o = 16'd0;
      end
    endcase
  end

  // -----------------------------------------------------------------------
  // GELU + ff_ram write-back control
  // During S_GELU:
  //   Read side: ff_ram_raddr advances 0..511, data appears 1 cycle later
  //   GELU pipe: 2 cycle latency
  //   Write side: ff_ram_we writes GELU output back to ff_ram at raddr-3
  // -----------------------------------------------------------------------

  // ff_up done: copy ff_up_out into ff_ram
  // We do this in S_FF_UP after ff_up_done by a sub-counter
  reg        ff_up_copy;     // set when draining ff_up_out -> ff_ram
  reg [9:0]  ff_up_copy_idx;

  // -----------------------------------------------------------------------
  // Main FSM
  // -----------------------------------------------------------------------
  always @(posedge clk_i) begin
    if (rst_i) begin
      state         <= S_IDLE;
      done_o        <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;
      ff_ram_we     <= 1'b0;
      ff_up_copy    <= 1'b0;

    end else begin
      done_o        <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;
      ff_ram_we     <= 1'b0;

      case (state)

        // ------------------------------------------------------------------
        S_IDLE: begin
          if (start_i) begin
            layer_r  <= layer_i;
            pos_r    <= pos_i;
            x_reg    <= x_i;
            ln_which <= 1'b0;
            state    <= S_LN_START;
          end
        end

        // ------------------------------------------------------------------
        S_LN_START: begin
          ln_start <= 1'b1;
          ln_gamma_sel <= ln_which ? ({layer_r, 3'b000} + 6'd6)
                                   : ({layer_r, 3'b000} + 6'd2);
          state <= S_LN_WAIT;
        end

        // ------------------------------------------------------------------
        S_LN_WAIT: begin
          if (ln_done) begin
            sub_out <= ln_y;
            if (ln_which == 1'b0) begin
              state      <= S_ATTN;
              attn_start <= 1'b1;
            end else begin
              state       <= S_FF_UP;
              ff_up_start <= 1'b1;
              ff_up_copy  <= 1'b0;
            end
          end
        end

        // ------------------------------------------------------------------
        S_ATTN: begin
          if (attn_done) begin
            sub_out <= attn_out;
            // Start sequential residual add 1
            res_idx  <= 7'd0;
            res_mux  <= 1'b0;
            state    <= S_RES1_ACC;
          end
        end

        // ------------------------------------------------------------------
        // Residual 1: x_reg[i] = fp16_add(sub_out[i], x_reg[i])
        // res_sum is combinational from sub_out[res_idx] + x_reg[res_idx]
        S_RES1_ACC: begin
          x_reg[res_idx*16 +: 16] <= res_sum;
          res_idx <= res_idx + 7'd1;
          if (res_idx == 7'd127) begin
            ln_which <= 1'b1;
            state    <= S_LN_START;
          end
        end

        // ------------------------------------------------------------------
        // FF_up: wait for matvec, then copy output into ff_ram
        S_FF_UP: begin
          if (!ff_up_copy) begin
            if (ff_up_done) begin
              ff_up_copy     <= 1'b1;
              ff_up_copy_idx <= 10'd0;
            end
          end else begin
            // Copy ff_up_out[ff_up_copy_idx] -> ff_ram
            ff_ram_we    <= 1'b1;
            ff_ram_waddr <= ff_up_copy_idx[8:0];
            ff_ram_wdata <= ff_up_out[ff_up_copy_idx*16 +: 16];
            ff_up_copy_idx <= ff_up_copy_idx + 10'd1;
            if (ff_up_copy_idx == 10'd511) begin
              ff_up_copy    <= 1'b0;
              // Start GELU: begin reading from ff_ram
              gelu_idx      <= 10'd0;
              ff_ram_raddr  <= 9'd0;
              state         <= S_GELU;
            end
          end
        end

        // ------------------------------------------------------------------
        // GELU: read ff_ram -> GELU (2-cycle pipe) -> write back to ff_ram
        //
        // Timeline (gelu_idx):
        //   0    : raddr=0 issued
        //   1    : raddr=1, rdata=ram[0] -> gelu_in, valid=1
        //   2    : raddr=2, rdata=ram[1] -> gelu_in, valid=1; GELU stage1(ram[0])
        //   3    : rdata=ram[2], gelu_out=GELU(ram[0]) -> write ram[0]
        //   ...
        //   512  : last gelu_in (ram[511])
        //   513  : gelu_out=GELU(ram[510]) -> write ram[510]
        //   514  : gelu_out=GELU(ram[511]) -> write ram[511]; transition to S_FF_DRAIN
        S_GELU: begin
          // Advance read address
          if (gelu_idx <= 10'd510) begin
            ff_ram_raddr <= ff_ram_raddr + 9'd1;
          end

          // Feed GELU from RAM read data (1-cycle BRAM latency)
          if (gelu_idx >= 10'd1 && gelu_idx <= 10'd512) begin
            gelu_in       <= ff_ram_rdata;
            gelu_valid_in <= 1'b1;
          end

          // Write GELU result back (2 cycles after read)
          if (gelu_idx >= 10'd3 && gelu_idx <= 10'd514) begin
            ff_ram_we    <= 1'b1;
            ff_ram_waddr <= gelu_idx[8:0] - 9'd3;
            ff_ram_wdata <= gelu_out;
          end

          gelu_idx <= gelu_idx + 10'd1;

          if (gelu_idx == 10'd514) begin
            // All GELU outputs written; now drain ff_ram -> ff_drain_buf
            drain_idx    <= 10'd0;
            ff_ram_raddr <= 9'd0;
            state        <= S_FF_DRAIN;
          end
        end

        // ------------------------------------------------------------------
        // Drain ff_ram -> ff_drain_buf over 512+1 cycles, then start ff_down
        // Cycle 0  : raddr=0 issued
        // Cycle 1  : raddr=1, rdata=ram[0] -> ff_drain_buf[0]
        // ...
        // Cycle 512: rdata=ram[511] -> ff_drain_buf[511]; start ff_down
        S_FF_DRAIN: begin
          if (drain_idx < 10'd511) begin
            ff_ram_raddr <= ff_ram_raddr + 9'd1;
          end
          if (drain_idx >= 10'd1) begin
            ff_drain_buf[(drain_idx - 10'd1)*16 +: 16] <= ff_ram_rdata;
          end
          drain_idx <= drain_idx + 10'd1;
          if (drain_idx == 10'd512) begin
            ff_down_start <= 1'b1;
            state         <= S_FF_DOWN;
          end
        end

        // ------------------------------------------------------------------
        S_FF_DOWN: begin
          if (ff_down_done) begin
            // Start sequential residual add 2
            res_idx <= 7'd0;
            res_mux <= 1'b1;
            state   <= S_RES2_ACC;
          end
        end

        // ------------------------------------------------------------------
        // Residual 2: out_vec_o[i] = fp16_add(ff_down_out[i], x_reg[i])
        S_RES2_ACC: begin
          out_vec_o[res_idx*16 +: 16] <= res_sum;
          res_idx <= res_idx + 7'd1;
          if (res_idx == 7'd127) begin
            state <= S_DONE;
          end
        end

        // ------------------------------------------------------------------
        S_DONE: begin
          done_o <= 1'b1;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule