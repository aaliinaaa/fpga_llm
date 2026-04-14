// Single transformer block: LN1 -> Attention -> Residual -> LN2 -> FF -> Residual
//
// Optimisations:
//   - ff_buf (8192-bit) -> M10K ff_ram (512x16)
//   - gen_res1/res2 128 parallel adders -> single sequential adder
//   - attention now has streaming output (out_valid_o/out_data_o/out_addr_o)
//     captured directly into sub_out register element-by-element
//   - ff_up streaming output captured directly into ff_ram (no intermediate bus)
//   - ff_down streaming output used directly in residual add (no out_vec_o register)
//   - ff_drain_buf still needed as in_vec_i bus for ff_down matvec

module transformer_layer (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  input  wire [2047:0] x_i,

  output reg  [5:0]    w_sel_o,
  output reg  [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,

  output wire          k_we_o,
  output wire [15:0]   k_wdata_o,
  input  wire [15:0]   k_rdata_i,

  output wire          v_we_o,
  output wire [15:0]   v_wdata_o,
  input  wire [15:0]   v_rdata_i,

  output wire [1:0]    kv_layer_o,
  output wire [2:0]    kv_head_o,
  output wire [7:0]    kv_pos_o,
  output wire [3:0]    kv_dim_o,

  output reg  [2047:0] out_vec_o,
  output reg           done_o
);

  `include "weight_scales.vh"

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
  reg [1:0] layer_r;
  reg [7:0] pos_r;

  reg [2047:0] x_reg;    // residual
  reg [2047:0] sub_out;  // LN or attn output, feeds FF_up and residual1

  // -----------------------------------------------------------------------
  // ff_ram: M10K 512x16
  // -----------------------------------------------------------------------
  (* ramstyle = "M10K" *) reg [15:0] ff_ram [0:511];
  reg        ff_ram_we;
  reg [8:0]  ff_ram_waddr;
  reg [15:0] ff_ram_wdata;
  reg [8:0]  ff_ram_raddr;
  reg [15:0] ff_ram_rdata_r;
  wire [15:0] ff_ram_rdata = ff_ram_rdata_r;

  always @(posedge clk_i) begin
    if (ff_ram_we)
      ff_ram[ff_ram_waddr] <= ff_ram_wdata;
    ff_ram_rdata_r <= ff_ram[ff_ram_raddr];
  end

  // -----------------------------------------------------------------------
  // Single residual adder (sequential, 128 cycles)
  // -----------------------------------------------------------------------
  reg [6:0] res_idx;
  reg       res_mux;  // 0=res1 (sub_out+x_reg), 1=res2 (ff_down stream+x_reg)

  // res2: ff_down result arrives streaming — capture into a 16-bit reg
  reg [15:0] ff_down_elem;  // current streaming element from ff_down
  reg        ff_down_elem_valid;

  wire [15:0] res_a = res_mux ? ff_down_elem : sub_out[res_idx*16 +: 16];
  wire [15:0] res_b = x_reg[res_idx*16 +: 16];
  wire [15:0] res_sum;
  fp16_add_comb u_res_add (.a_i(res_a), .b_i(res_b), .sum_o(res_sum));

  reg ln_which;
  reg [9:0] gelu_idx;

  // Per-layer scales
  reg [15:0] ln_gamma_scale, ln_beta_scale, ff_up_scale, ff_down_scale;

  always @(*) begin
    case (layer_r)
      2'd0: begin ff_up_scale = SCALE_BLOCK0_FF_UP_WEIGHT; ff_down_scale = SCALE_BLOCK0_FF_DOWN_WEIGHT; end
      2'd1: begin ff_up_scale = SCALE_BLOCK1_FF_UP_WEIGHT; ff_down_scale = SCALE_BLOCK1_FF_DOWN_WEIGHT; end
      2'd2: begin ff_up_scale = SCALE_BLOCK2_FF_UP_WEIGHT; ff_down_scale = SCALE_BLOCK2_FF_DOWN_WEIGHT; end
      default: begin ff_up_scale = SCALE_BLOCK3_FF_UP_WEIGHT; ff_down_scale = SCALE_BLOCK3_FF_DOWN_WEIGHT; end
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
  reg         ln_start;
  reg  [5:0]  ln_gamma_sel;
  wire [5:0]  ln_w_sel;
  wire [6:0]  ln_w_addr;
  wire [2047:0] ln_y;
  wire        ln_done;

  layernorm u_ln (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ln_start),
    .x_i(x_reg),
    .w_sel_o(ln_w_sel), .w_addr_o(ln_w_addr), .w_data_i(w_data_i),
    .gamma_sel_i(ln_gamma_sel),
    .gamma_scale_i(ln_gamma_scale), .beta_scale_i(ln_beta_scale),
    .y_o(ln_y), .done_o(ln_done), .busy_o()
  );

  // -----------------------------------------------------------------------
  // Attention — streaming output captured into sub_out
  // -----------------------------------------------------------------------
  reg         attn_start;
  wire [5:0]  attn_w_sel;
  wire [15:0] attn_w_addr;
  wire        attn_out_valid;
  wire [15:0] attn_out_data;
  wire [6:0]  attn_out_addr;
  wire        attn_done;
  wire        attn_k_we, attn_v_we;
  wire [15:0] attn_k_wdata, attn_v_wdata;
  wire [1:0]  attn_kv_layer;
  wire [2:0]  attn_kv_head;
  wire [7:0]  attn_kv_pos;
  wire [3:0]  attn_kv_dim;

  attention u_attn (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(attn_start),
    .layer_i(layer_r), .pos_i(pos_r), .x_i(sub_out),
    .w_sel_o(attn_w_sel), .w_addr_o(attn_w_addr), .w_data_i(w_data_i),
    .k_we_o(attn_k_we), .k_wdata_o(attn_k_wdata), .k_rdata_i(k_rdata_i),
    .v_we_o(attn_v_we), .v_wdata_o(attn_v_wdata), .v_rdata_i(v_rdata_i),
    .kv_layer_o(attn_kv_layer), .kv_head_o(attn_kv_head),
    .kv_pos_o(attn_kv_pos), .kv_dim_o(attn_kv_dim),
    .out_valid_o(attn_out_valid), .out_data_o(attn_out_data),
    .out_addr_o(attn_out_addr),
    .done_o(attn_done)
  );

  assign k_we_o     = attn_k_we;
  assign k_wdata_o  = attn_k_wdata;
  assign v_we_o     = attn_v_we;
  assign v_wdata_o  = attn_v_wdata;
  assign kv_layer_o = attn_kv_layer;
  assign kv_head_o  = attn_kv_head;
  assign kv_pos_o   = attn_kv_pos;
  assign kv_dim_o   = attn_kv_dim;

  // Capture attention streaming output directly into sub_out
  always @(posedge clk_i) begin
    if (attn_out_valid)
      sub_out[attn_out_addr*16 +: 16] <= attn_out_data;
  end

  // -----------------------------------------------------------------------
  // FF_up: 128->512, streaming output -> ff_ram directly
  // -----------------------------------------------------------------------
  reg         ff_up_start;
  wire [15:0] ff_up_w_addr;
  wire        ff_up_out_valid;
  wire [15:0] ff_up_out_data;
  wire [8:0]  ff_up_out_addr;
  wire        ff_up_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(512)) u_ff_up (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ff_up_start),
    .in_vec_i(sub_out), .scale_i(ff_up_scale),
    .weight_addr_o(ff_up_w_addr), .weight_data_i(w_data_i),
    .out_valid_o(ff_up_out_valid), .out_data_o(ff_up_out_data),
    .out_addr_o(ff_up_out_addr),
    .done_o(ff_up_done)
  );

  // ff_up streams directly into ff_ram
  always @(*) begin
    if (ff_up_out_valid) begin
      ff_ram_we    = 1'b1;
      ff_ram_waddr = ff_up_out_addr;
      ff_ram_wdata = ff_up_out_data;
    end else if (state == S_GELU && gelu_idx >= 10'd3 && gelu_idx <= 10'd514) begin
      ff_ram_we    = 1'b1;
      ff_ram_waddr = gelu_idx[8:0] - 9'd3;
      ff_ram_wdata = gelu_out;
    end else begin
      ff_ram_we    = 1'b0;
      ff_ram_waddr = 9'd0;
      ff_ram_wdata = 16'd0;
    end
  end

  // -----------------------------------------------------------------------
  // FF_down: 512->128, streaming output used directly in RES2
  // in_vec_i must be full bus — filled from ff_ram over 512 cycles (S_FF_DRAIN)
  // -----------------------------------------------------------------------
  reg [8191:0] ff_drain_buf;
  reg [9:0]    drain_idx;

  wire [15:0]  ff_down_w_addr;
  wire         ff_down_out_valid;
  wire [15:0]  ff_down_out_data;
  wire [6:0]   ff_down_out_addr;
  wire         ff_down_done;
  reg          ff_down_start;

  matvec_fp16 #(.IN_DIM(512), .OUT_DIM(128)) u_ff_down (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ff_down_start),
    .in_vec_i(ff_drain_buf), .scale_i(ff_down_scale),
    .weight_addr_o(ff_down_w_addr), .weight_data_i(w_data_i),
    .out_valid_o(ff_down_out_valid), .out_data_o(ff_down_out_data),
    .out_addr_o(ff_down_out_addr),
    .done_o(ff_down_done)
  );

  // RES2: as ff_down streams results, immediately add x_reg and write out_vec_o
  // ff_down emits elements in order 0..127, one per ~IN_DIM cycles
  // We use a simple approach: when ff_down_out_valid, do the add and store
  wire [15:0] res2_b = x_reg[ff_down_out_addr*16 +: 16];
  wire [15:0] res2_sum;
  fp16_add_comb u_res2_add (.a_i(ff_down_out_data), .b_i(res2_b), .sum_o(res2_sum));

  always @(posedge clk_i) begin
    if (ff_down_out_valid)
      out_vec_o[ff_down_out_addr*16 +: 16] <= res2_sum;
  end

  // -----------------------------------------------------------------------
  // GELU
  // -----------------------------------------------------------------------
  reg         gelu_valid_in;
  reg  [15:0] gelu_in;
  wire [15:0] gelu_out;

  gelu u_gelu (
    .clk_i(clk_i), .valid_i(gelu_valid_in),
    .x_i(gelu_in), .valid_o(), .y_o(gelu_out)
  );

  // -----------------------------------------------------------------------
  // Weight store mux
  // -----------------------------------------------------------------------
  always @(*) begin
    case (state)
      S_LN_START, S_LN_WAIT: begin w_sel_o = ln_w_sel;  w_addr_o = {9'd0, ln_w_addr}; end
      S_ATTN:    begin w_sel_o = attn_w_sel; w_addr_o = attn_w_addr; end
      S_FF_UP:   begin w_sel_o = {layer_r, 3'b000} + 6'd8; w_addr_o = ff_up_w_addr; end
      S_FF_DOWN: begin w_sel_o = {layer_r, 3'b000} + 6'd9; w_addr_o = ff_down_w_addr; end
      default:   begin w_sel_o = 6'd0; w_addr_o = 16'd0; end
    endcase
  end

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

    end else begin
      done_o        <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            layer_r  <= layer_i;
            pos_r    <= pos_i;
            x_reg    <= x_i;
            ln_which <= 1'b0;
            state    <= S_LN_START;
          end
        end

        S_LN_START: begin
          ln_start     <= 1'b1;
          ln_gamma_sel <= ln_which ? ({layer_r, 3'b000} + 6'd6)
                                   : ({layer_r, 3'b000} + 6'd2);
          state <= S_LN_WAIT;
        end

        S_LN_WAIT: begin
          if (ln_done) begin
            //sub_out <= ln_y;
            if (ln_which == 1'b0) begin
              state      <= S_ATTN;
              attn_start <= 1'b1;
            end else begin
              state       <= S_FF_UP;
              ff_up_start <= 1'b1;
            end
          end
        end

        // Attention: streaming output captured into sub_out by always block above
        S_ATTN: begin
          if (attn_done) begin
            // sub_out is already populated element-by-element during attn run
            res_idx <= 7'd0;
            res_mux <= 1'b0;
            state   <= S_RES1_ACC;
          end
        end

        // Residual 1: x_reg[i] = sub_out[i] + x_reg[i]
        S_RES1_ACC: begin
          x_reg[res_idx*16 +: 16] <= res_sum;
          res_idx <= res_idx + 7'd1;
          if (res_idx == 7'd127) begin
            ln_which <= 1'b1;
            state    <= S_LN_START;
          end
        end

        // FF_up: streaming output goes directly into ff_ram via always @(*)
        S_FF_UP: begin
          if (ff_up_done) begin
            gelu_idx     <= 10'd0;
            ff_ram_raddr <= 9'd0;
            state        <= S_GELU;
          end
        end

        // GELU: read ff_ram -> GELU pipe -> write back ff_ram
        // ff_ram write is handled in always @(*) above
        S_GELU: begin
          if (gelu_idx <= 10'd510)
            ff_ram_raddr <= ff_ram_raddr + 9'd1;

          if (gelu_idx >= 10'd1 && gelu_idx <= 10'd512) begin
            gelu_in       <= ff_ram_rdata;
            gelu_valid_in <= 1'b1;
          end

          gelu_idx <= gelu_idx + 10'd1;

          if (gelu_idx == 10'd514) begin
            drain_idx    <= 10'd0;
            ff_ram_raddr <= 9'd0;
            state        <= S_FF_DRAIN;
          end
        end

        // Drain ff_ram -> ff_drain_buf (512 cycles), then start ff_down
        S_FF_DRAIN: begin
          if (drain_idx < 10'd511)
            ff_ram_raddr <= ff_ram_raddr + 9'd1;
          if (drain_idx >= 10'd1)
            ff_drain_buf[(drain_idx - 10'd1)*16 +: 16] <= ff_ram_rdata;
          drain_idx <= drain_idx + 10'd1;
          if (drain_idx == 10'd512) begin
            ff_down_start <= 1'b1;
            state         <= S_FF_DOWN;
          end
        end

        // FF_down: streaming output + residual handled by always blocks above
        // Wait for done_o — out_vec_o fills up as elements stream in
        S_FF_DOWN: begin
          if (ff_down_done) begin
            done_o <= 1'b1;
            state  <= S_IDLE;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
