// Single transformer block: LN1 -> Attention -> Residual -> LN2 -> FF -> Residual
//
// Key changes vs v2:
//   ff_drain_buf [8191:0] ELIMINATED — ff_ram read by u_ff_down via col_addr_o
//   sub_ram M10K 128x16  — u_ff_up reads via col_addr_o (no sub_out wide bus for matvec)
//   sub_out_flat [2047:0] still needed for attention x_i input
//   x_reg [2047:0] still needed for layernorm x_i and residual

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

  output wire          k_we_o,   output wire [15:0] k_wdata_o,  input wire [15:0] k_rdata_i,
  output wire          v_we_o,   output wire [15:0] v_wdata_o,  input wire [15:0] v_rdata_i,
  output wire [1:0]    kv_layer_o, output wire [2:0] kv_head_o,
  output wire [7:0]    kv_pos_o,   output wire [3:0] kv_dim_o,

  output reg  [2047:0] out_vec_o,
  output reg           done_o
);

  `include "weight_scales.vh"

  localparam [3:0]
    S_IDLE     = 4'd0, S_LN_START = 4'd1, S_LN_WAIT  = 4'd2,
    S_LN_COPY  = 4'd11,                   // copy ln_y -> sub_ram + sub_out_flat
    S_ATTN     = 4'd3, S_RES1_PRE = 4'd12,// pre-issue sub_ram[0]
    S_RES1_ACC = 4'd4,
    S_FF_UP    = 4'd5, S_GELU     = 4'd6,
    S_FF_DOWN  = 4'd7, S_DONE     = 4'd9;

  reg [3:0] state;
  reg [1:0] layer_r;
  reg [7:0] pos_r;
  reg [2047:0] x_reg;
  reg [2047:0] sub_out_flat;  // for attention x_i
  reg ln_which;

  // -----------------------------------------------------------------------
  // sub_ram: M10K 128x16 — source for u_ff_up
  // -----------------------------------------------------------------------
  (* ramstyle = "M10K" *) reg [15:0] sub_ram [0:127];
  reg        sub_ram_we;
  reg [6:0]  sub_ram_waddr;
  reg [15:0] sub_ram_wdata;
  wire [6:0] sub_ram_raddr;   // driven by ff_up col_addr_o or res_idx
  reg [15:0] sub_ram_rdata_r;

  always @(posedge clk_i) begin
    if (sub_ram_we) sub_ram[sub_ram_waddr] <= sub_ram_wdata;
    sub_ram_rdata_r <= sub_ram[sub_ram_raddr];
  end

  // -----------------------------------------------------------------------
  // ff_ram: M10K 512x16 — source for u_ff_down (via col_addr_o)
  // -----------------------------------------------------------------------
  (* ramstyle = "M10K" *) reg [15:0] ff_ram [0:511];
  reg        ff_ram_we;
  reg [8:0]  ff_ram_waddr;
  reg [15:0] ff_ram_wdata;
  wire [8:0] ff_ram_raddr;    // driven by ff_down col_addr_o or GELU read
  reg [15:0] ff_ram_rdata_r;

  always @(posedge clk_i) begin
    if (ff_ram_we) ff_ram[ff_ram_waddr] <= ff_ram_wdata;
    ff_ram_rdata_r <= ff_ram[ff_ram_raddr];
  end

  // -----------------------------------------------------------------------
  // Residual 1 counter
  // -----------------------------------------------------------------------
  reg [7:0] res_idx;  // 0..129 (extra range for pipeline drain)

  wire [15:0] res1_sum;
  fp16_add_comb u_res_add (
    .a_i(sub_ram_rdata_r),
    .b_i(x_reg[(res_idx > 8'd0 ? res_idx - 8'd1 : 8'd0)*16 +: 16]),
    .sum_o(res1_sum)
  );

  // -----------------------------------------------------------------------
  // Per-layer scales
  // -----------------------------------------------------------------------
  reg [15:0] ln_gamma_scale, ln_beta_scale, ff_up_scale, ff_down_scale;
  always @(*) begin
    case (layer_r)
      2'd0: begin ff_up_scale=SCALE_BLOCK0_FF_UP_WEIGHT; ff_down_scale=SCALE_BLOCK0_FF_DOWN_WEIGHT; end
      2'd1: begin ff_up_scale=SCALE_BLOCK1_FF_UP_WEIGHT; ff_down_scale=SCALE_BLOCK1_FF_DOWN_WEIGHT; end
      2'd2: begin ff_up_scale=SCALE_BLOCK2_FF_UP_WEIGHT; ff_down_scale=SCALE_BLOCK2_FF_DOWN_WEIGHT; end
      default: begin ff_up_scale=SCALE_BLOCK3_FF_UP_WEIGHT; ff_down_scale=SCALE_BLOCK3_FF_DOWN_WEIGHT; end
    endcase
  end
  always @(*) begin
    case ({layer_r, ln_which})
      3'b000: begin ln_gamma_scale=SCALE_BLOCK0_LN1_WEIGHT; ln_beta_scale=SCALE_BLOCK0_LN1_BIAS; end
      3'b001: begin ln_gamma_scale=SCALE_BLOCK0_LN2_WEIGHT; ln_beta_scale=SCALE_BLOCK0_LN2_BIAS; end
      3'b010: begin ln_gamma_scale=SCALE_BLOCK1_LN1_WEIGHT; ln_beta_scale=SCALE_BLOCK1_LN1_BIAS; end
      3'b011: begin ln_gamma_scale=SCALE_BLOCK1_LN2_WEIGHT; ln_beta_scale=SCALE_BLOCK1_LN2_BIAS; end
      3'b100: begin ln_gamma_scale=SCALE_BLOCK2_LN1_WEIGHT; ln_beta_scale=SCALE_BLOCK2_LN1_BIAS; end
      3'b101: begin ln_gamma_scale=SCALE_BLOCK2_LN2_WEIGHT; ln_beta_scale=SCALE_BLOCK2_LN2_BIAS; end
      3'b110: begin ln_gamma_scale=SCALE_BLOCK3_LN1_WEIGHT; ln_beta_scale=SCALE_BLOCK3_LN1_BIAS; end
      default: begin ln_gamma_scale=SCALE_BLOCK3_LN2_WEIGHT; ln_beta_scale=SCALE_BLOCK3_LN2_BIAS; end
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
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ln_start), .x_i(x_reg),
    .w_sel_o(ln_w_sel), .w_addr_o(ln_w_addr), .w_data_i(w_data_i),
    .gamma_sel_i(ln_gamma_sel), .gamma_scale_i(ln_gamma_scale),
    .beta_scale_i(ln_beta_scale), .y_o(ln_y), .done_o(ln_done), .busy_o()
  );

  // -----------------------------------------------------------------------
  // Attention — streaming output updates sub_ram and sub_out_flat
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
  wire [1:0]  attn_kv_layer;  wire [2:0] attn_kv_head;
  wire [7:0]  attn_kv_pos;    wire [3:0] attn_kv_dim;

  attention u_attn (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(attn_start),
    .layer_i(layer_r), .pos_i(pos_r), .x_i(sub_out_flat),
    .w_sel_o(attn_w_sel), .w_addr_o(attn_w_addr), .w_data_i(w_data_i),
    .k_we_o(attn_k_we), .k_wdata_o(attn_k_wdata), .k_rdata_i(k_rdata_i),
    .v_we_o(attn_v_we), .v_wdata_o(attn_v_wdata), .v_rdata_i(v_rdata_i),
    .kv_layer_o(attn_kv_layer), .kv_head_o(attn_kv_head),
    .kv_pos_o(attn_kv_pos), .kv_dim_o(attn_kv_dim),
    .out_valid_o(attn_out_valid), .out_data_o(attn_out_data),
    .out_addr_o(attn_out_addr), .done_o(attn_done)
  );

  assign k_we_o=attn_k_we; assign k_wdata_o=attn_k_wdata;
  assign v_we_o=attn_v_we; assign v_wdata_o=attn_v_wdata;
  assign kv_layer_o=attn_kv_layer; assign kv_head_o=attn_kv_head;
  assign kv_pos_o=attn_kv_pos;     assign kv_dim_o=attn_kv_dim;

  always @(posedge clk_i) begin
    if (attn_out_valid)
      sub_out_flat[attn_out_addr*16 +: 16] <= attn_out_data;
  end

  // -----------------------------------------------------------------------
  // FF_up: col_addr_o -> sub_ram, result -> ff_ram
  // -----------------------------------------------------------------------
  reg         ff_up_start;
  wire [6:0]  ff_up_col_addr;
  wire [14:0] ff_up_w_addr;
  wire        ff_up_out_valid;
  wire [15:0] ff_up_out_data;
  wire [8:0]  ff_up_out_addr;
  wire        ff_up_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(512)) u_ff_up (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ff_up_start),
    .col_addr_o(ff_up_col_addr), .col_data_i(sub_ram_rdata_r),
    .scale_i(ff_up_scale),
    .weight_addr_o(ff_up_w_addr), .weight_data_i(w_data_i),
    .out_valid_o(ff_up_out_valid), .out_data_o(ff_up_out_data),
    .out_addr_o(ff_up_out_addr), .done_o(ff_up_done)
  );

  // sub_ram read address mux: ff_up or res1
  assign sub_ram_raddr = (state == S_FF_UP) ? ff_up_col_addr : res_idx[6:0];

  // -----------------------------------------------------------------------
  // GELU state
  // -----------------------------------------------------------------------
  reg [9:0]  gelu_ridx;
  reg [8:0]  gelu_raddr_r;
  reg        gelu_valid_in;
  reg [15:0] gelu_in;
  wire [15:0] gelu_out;
  wire        gelu_valid_out;

  gelu u_gelu (.clk_i(clk_i), .valid_i(gelu_valid_in),
               .x_i(gelu_in), .valid_o(gelu_valid_out), .y_o(gelu_out));

  // -----------------------------------------------------------------------
  // FF_down: col_addr_o -> ff_ram, result -> residual2 -> out_vec_o
  // -----------------------------------------------------------------------
  reg         ff_down_start;
  wire [8:0]  ff_down_col_addr;
  wire [15:0] ff_down_w_addr;
  wire        ff_down_out_valid;
  wire [15:0] ff_down_out_data;
  wire [6:0]  ff_down_out_addr;
  wire        ff_down_done;

  matvec_fp16 #(.IN_DIM(512), .OUT_DIM(128)) u_ff_down (
    .clk_i(clk_i), .rst_i(rst_i), .start_i(ff_down_start),
    .col_addr_o(ff_down_col_addr), .col_data_i(ff_ram_rdata_r),
    .scale_i(ff_down_scale),
    .weight_addr_o(ff_down_w_addr), .weight_data_i(w_data_i),
    .out_valid_o(ff_down_out_valid), .out_data_o(ff_down_out_data),
    .out_addr_o(ff_down_out_addr), .done_o(ff_down_done)
  );

  // ff_ram read address mux: ff_down or GELU
  assign ff_ram_raddr = (state == S_FF_DOWN) ? ff_down_col_addr : gelu_raddr_r;

  // RES2: immediate add when ff_down emits
  wire [15:0] res2_b = x_reg[ff_down_out_addr*16 +: 16];
  wire [15:0] res2_sum;
  fp16_add_comb u_res2_add (.a_i(ff_down_out_data), .b_i(res2_b), .sum_o(res2_sum));

  always @(posedge clk_i) begin
    if (ff_down_out_valid)
      out_vec_o[ff_down_out_addr*16 +: 16] <= res2_sum;
  end

  // -----------------------------------------------------------------------
  // ff_ram write mux: ff_up or GELU writeback
  // -----------------------------------------------------------------------
  always @(*) begin
    if (ff_up_out_valid) begin
      ff_ram_we = 1'b1; ff_ram_waddr = ff_up_out_addr; ff_ram_wdata = ff_up_out_data;
    end else if (gelu_valid_out && state == S_GELU) begin
      ff_ram_we = 1'b1;
      ff_ram_waddr = (gelu_ridx >= 10'd2) ? gelu_ridx[8:0] - 9'd2 : 9'd0;
      ff_ram_wdata = gelu_out;
    end else begin
      ff_ram_we = 1'b0; ff_ram_waddr = 9'd0; ff_ram_wdata = 16'd0;
    end
  end

  // -----------------------------------------------------------------------
  // sub_ram write mux: ln_copy or attn streaming
  // -----------------------------------------------------------------------
  reg        ln_copy_active;
  reg [6:0]  ln_copy_idx;

  always @(*) begin
    if (attn_out_valid) begin
      sub_ram_we = 1'b1; sub_ram_waddr = attn_out_addr; sub_ram_wdata = attn_out_data;
    end else if (ln_copy_active) begin
      sub_ram_we = 1'b1; sub_ram_waddr = ln_copy_idx;
      sub_ram_wdata = ln_y[ln_copy_idx*16 +: 16];
    end else begin
      sub_ram_we = 1'b0; sub_ram_waddr = 7'd0; sub_ram_wdata = 16'd0;
    end
  end

  // -----------------------------------------------------------------------
  // Weight store mux
  // -----------------------------------------------------------------------
  always @(*) begin
    case (state)
      S_LN_START, S_LN_WAIT, S_LN_COPY:
                 begin w_sel_o=ln_w_sel; w_addr_o={9'd0,ln_w_addr}; end
      S_ATTN:    begin w_sel_o=attn_w_sel; w_addr_o=attn_w_addr; end
      S_FF_UP:   begin w_sel_o={layer_r,3'b000}+6'd8; w_addr_o={1'b0,ff_up_w_addr}; end
      S_FF_DOWN: begin w_sel_o={layer_r,3'b000}+6'd9; w_addr_o=ff_down_w_addr; end
      default:   begin w_sel_o=6'd0; w_addr_o=16'd0; end
    endcase
  end

  // -----------------------------------------------------------------------
  // Main FSM
  // -----------------------------------------------------------------------
  always @(posedge clk_i) begin
    if (rst_i) begin
      state <= S_IDLE; done_o <= 1'b0;
      ln_start <= 1'b0; attn_start <= 1'b0;
      ff_up_start <= 1'b0; ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;
      ln_copy_active <= 1'b0;

    end else begin
      done_o <= 1'b0; ln_start <= 1'b0; attn_start <= 1'b0;
      ff_up_start <= 1'b0; ff_down_start <= 1'b0; gelu_valid_in <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            layer_r <= layer_i; pos_r <= pos_i; x_reg <= x_i;
            ln_which <= 1'b0; state <= S_LN_START;
          end
        end

        S_LN_START: begin
          ln_start <= 1'b1;
          ln_gamma_sel <= ln_which ? ({layer_r,3'b000}+6'd6) : ({layer_r,3'b000}+6'd2);
          state <= S_LN_WAIT;
        end

        S_LN_WAIT: begin
          if (ln_done) begin
            ln_copy_active <= 1'b1;
            ln_copy_idx    <= 7'd0;
            state          <= S_LN_COPY;
          end
        end

        // Copy ln_y -> sub_ram + sub_out_flat (128 cycles)
        S_LN_COPY: begin
        //   sub_out_flat[ln_copy_idx*16 +: 16] <= ln_y[ln_copy_idx*16 +: 16];
          ln_copy_idx <= ln_copy_idx + 7'd1;
          if (ln_copy_idx == 7'd127) begin
            ln_copy_active <= 1'b0;
            if (ln_which == 1'b0) begin
              attn_start <= 1'b1; state <= S_ATTN;
            end else begin
              ff_up_start <= 1'b1; state <= S_FF_UP;
            end
          end
        end

        S_ATTN: begin
          if (attn_done) begin
            // attn has been writing sub_ram during its run
            // Pre-read sub_ram[0] for residual
            res_idx <= 8'd0;
            state   <= S_RES1_PRE;
          end
        end

        // Pre-issue sub_ram read (1-cycle latency)
        S_RES1_PRE: begin
          res_idx <= 8'd1;
          state   <= S_RES1_ACC;
        end

        // Residual 1: x_reg[i] = sub_ram[i] + x_reg[i]
        // sub_ram_rdata_r has data for res_idx-1
        S_RES1_ACC: begin
          if (res_idx > 8'd0)
            x_reg[(res_idx-8'd1)*16 +: 16] <= res1_sum;
          res_idx <= res_idx + 8'd1;
          if (res_idx == 8'd128) begin
            // Last element (idx=128 -> data for idx=127 arrives next cycle)
            // Wait one more cycle
            state <= S_LN_START;  // will execute after last element written
            x_reg[127*16 +: 16] <= res1_sum;
            ln_which <= 1'b1;
          end
        end

        S_FF_UP: begin
          if (ff_up_done) begin
            gelu_ridx    <= 10'd0;
            gelu_raddr_r <= 9'd0;
            state        <= S_GELU;
          end
        end

        S_GELU: begin
          if (gelu_ridx <= 10'd510)
            gelu_raddr_r <= gelu_raddr_r + 9'd1;

          if (gelu_ridx >= 10'd1 && gelu_ridx <= 10'd512) begin
            gelu_in       <= ff_ram_rdata_r;
            gelu_valid_in <= 1'b1;
          end

          gelu_ridx <= gelu_ridx + 10'd1;

          if (gelu_ridx == 10'd514) begin
            ff_down_start <= 1'b1;
            state         <= S_FF_DOWN;
          end
        end

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
