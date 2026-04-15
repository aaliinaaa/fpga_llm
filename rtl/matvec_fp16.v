// Matrix-vector multiply: int8 weights x fp16 input -> fp16 output
//
// Streaming INPUT interface (eliminates wide in_vec_i bus and its fanout):
//   in_valid_i  — caller asserts for IN_DIM consecutive cycles
//   in_data_i   — fp16 input element for current column
//   in_ready_o  — asserted when matvec is ready to accept input
//
// Streaming OUTPUT interface (no out_vec_o register):
//   out_valid_o — pulses 1 cycle when a row is complete
//   out_data_o  — fp16 result for that row
//   out_addr_o  — row index (0..OUT_DIM-1)
//
// Protocol:
//   1. Assert start_i for 1 cycle
//   2. For each of OUT_DIM rows, stream IN_DIM elements via in_valid_i/in_data_i
//      (matvec re-uses the same input stream for each row — caller must
//       provide the full vector once per row, i.e. IN_DIM*OUT_DIM total elements
//       in row-major order matching the weight matrix)
//
// WAIT — simpler protocol: caller streams the input vector ONCE.
// matvec internally cycles through all OUT_DIM rows, requesting the
// input vector OUT_DIM times. We use an internal column counter and
// req_o / ack handshake:
//
// Actually simplest: caller provides input from a RAM with address driven
// by matvec. We expose col_addr_o so caller can read from its RAM.
//
// Interface:
//   col_addr_o [$clog2(IN_DIM)-1:0] — column address matvec wants to read
//   col_data_i [15:0]               — input data from caller's RAM (1-cycle latency)
//   col_req_o                       — asserted when col_addr_o is valid
//
// The caller connects col_addr_o to its RAM read address and col_data_i
// to the RAM read data (registered). matvec handles the 1-cycle latency.
//
// Weight interface unchanged: weight_addr_o / weight_data_i (BRAM, 1-cycle latency)
//
// Latency: OUT_DIM * (IN_DIM + 1) + 2 cycles  (extra cycle per row for latency)

module matvec_fp16 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire                              clk_i,
  input  wire                              rst_i,
  input  wire                              start_i,

  // Input vector via RAM interface (replaces wide in_vec_i bus)
  output reg  [$clog2(IN_DIM)-1:0]         col_addr_o,   // column to read
  input  wire [15:0]                        col_data_i,   // data (1-cycle latency)

  input  wire [15:0]                        scale_i,
  output reg  [$clog2(OUT_DIM*IN_DIM)-1:0] weight_addr_o,
  input  wire signed [7:0]                  weight_data_i,

  // Streaming output
  output reg                                out_valid_o,
  output reg  [15:0]                        out_data_o,
  output reg  [$clog2(OUT_DIM)-1:0]         out_addr_o,

  output reg                                done_o
);

  localparam ADDR_W  = $clog2(OUT_DIM * IN_DIM);
  localparam COL_W   = $clog2(IN_DIM);
  localparam ROW_W   = $clog2(OUT_DIM);

  // Dequant: int8 -> fp16 -> * scale
  wire [15:0] w_fp16;
  fp16_from_int8 u_dequant_cvt (.val_i(weight_data_i), .fp16_o(w_fp16));

  wire [15:0] w_dequant;
  fp16_mul_comb u_dequant_mul (.a_i(w_fp16), .b_i(scale_i), .prod_o(w_dequant));

  // MAC: w_dequant * col_data (1-cycle latency from col_addr_o)
  wire [15:0] mac_prod;
  fp16_mul_comb u_mac_mul (.a_i(w_dequant), .b_i(col_data_i), .prod_o(mac_prod));

  reg [15:0] acc;
  wire [15:0] acc_sum;
  fp16_add_comb u_acc_add (.a_i(acc), .b_i(mac_prod), .sum_o(acc_sum));

  reg [COL_W:0]  col;   // current column (extra bit for overflow detection)
  reg [ROW_W:0]  row;
  reg            running;
  reg            prefetch;

  // col_addr_o advances 1 cycle ahead (account for RAM read latency)
  // weight_addr_o also advances 1 cycle ahead
  // Both are issued simultaneously so data arrives aligned

  always @(posedge clk_i) begin
    if (rst_i) begin
      col           <= {(COL_W+1){1'b0}};
      row           <= {(ROW_W+1){1'b0}};
      acc           <= 16'd0;
      running       <= 1'b0;
      prefetch      <= 1'b0;
      done_o        <= 1'b0;
      out_valid_o   <= 1'b0;
      out_data_o    <= 16'd0;
      out_addr_o    <= {ROW_W{1'b0}};
      weight_addr_o <= {ADDR_W{1'b0}};
      col_addr_o    <= {COL_W{1'b0}};

    end else if (start_i) begin
      col           <= {(COL_W+1){1'b0}};
      row           <= {(ROW_W+1){1'b0}};
      acc           <= 16'd0;
      running       <= 1'b0;
      prefetch      <= 1'b1;
      done_o        <= 1'b0;
      out_valid_o   <= 1'b0;
      weight_addr_o <= {ADDR_W{1'b0}};
      col_addr_o    <= {COL_W{1'b0}};

    end else if (prefetch) begin
      // Issue first addresses, wait 1 cycle for data
      prefetch      <= 1'b0;
      running       <= 1'b1;
      weight_addr_o <= weight_addr_o + 1;
      col_addr_o    <= col_addr_o + 1;  // will wrap to 0 at start of row
      out_valid_o   <= 1'b0;

    end else if (running) begin
      out_valid_o <= 1'b0;
      done_o      <= 1'b0;

      if (col == IN_DIM[COL_W:0] - 1) begin
        // Last element of this row — output result
        out_valid_o <= 1'b1;
        out_data_o  <= acc_sum;
        out_addr_o  <= row[ROW_W-1:0];

        acc <= 16'd0;
        col <= {(COL_W+1){1'b0}};

        // Next row: reset col_addr to 0
        col_addr_o    <= {COL_W{1'b0}};
        weight_addr_o <= weight_addr_o + 1;
        row           <= row + 1;

        if (row == OUT_DIM[ROW_W:0] - 1) begin
          running <= 1'b0;
          done_o  <= 1'b1;
        end

      end else begin
        acc           <= acc_sum;
        col           <= col + 1;
        col_addr_o    <= col[COL_W-1:0] + 1;  // issue next col address
        weight_addr_o <= weight_addr_o + 1;
      end

    end else begin
      done_o      <= 1'b0;
      out_valid_o <= 1'b0;
    end
  end

endmodule


module fp16_add_comb (
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  output wire [15:0] sum_o
);

  wire        a_sign = a_i[15];
  wire [4:0]  a_exp  = a_i[14:10];
  wire [9:0]  a_mant = a_i[9:0];
  wire        b_sign = b_i[15];
  wire [4:0]  b_exp  = b_i[14:10];
  wire [9:0]  b_mant = b_i[9:0];

  wire a_is_zero = (a_exp == 5'd0);
  wire b_is_zero = (b_exp == 5'd0);
  wire a_is_inf  = (a_exp == 5'd31) && (a_mant == 10'd0);
  wire b_is_inf  = (b_exp == 5'd31) && (b_mant == 10'd0);
  wire a_is_nan  = (a_exp == 5'd31) && (a_mant != 10'd0);
  wire b_is_nan  = (b_exp == 5'd31) && (b_mant != 10'd0);

  wire [10:0] a_full = a_is_zero ? 11'd0 : {1'b1, a_mant};
  wire [10:0] b_full = b_is_zero ? 11'd0 : {1'b1, b_mant};

  wire a_ge_b = (a_exp > b_exp) || ((a_exp == b_exp) && (a_full >= b_full));

  wire        lg_sign = a_ge_b ? a_sign : b_sign;
  wire [4:0]  lg_exp  = a_ge_b ? a_exp  : b_exp;
  wire [10:0] lg_mant = a_ge_b ? a_full : b_full;
  wire        sm_sign = a_ge_b ? b_sign : a_sign;
  wire [10:0] sm_mant = a_ge_b ? b_full : a_full;
  wire [4:0]  sm_exp  = a_ge_b ? b_exp  : a_exp;

  wire [4:0] exp_diff = lg_exp - sm_exp;
  wire [13:0] lg_ext = {1'b0, lg_mant, 2'b00};
  wire [26:0] sm_wide = {1'b0, sm_mant, 2'b00, 13'b0};
  wire [26:0] sm_shifted = sm_wide >> exp_diff;
  wire [13:0] sm_ext = sm_shifted[26:13];
  wire        sticky  = |sm_shifted[12:0];

  wire eff_sub = lg_sign ^ sm_sign;
  wire [14:0] mant_sum = eff_sub ? ({1'b0, lg_ext} - {1'b0, sm_ext}) :
                                   ({1'b0, lg_ext} + {1'b0, sm_ext});

  reg [3:0] lod;
  reg       sum_is_zero;
  integer i;
  always @(*) begin
    lod = 4'd0;
    sum_is_zero = (mant_sum[14:0] == 15'd0);
    for (i = 0; i < 15; i = i + 1) begin
      if (mant_sum[i]) lod = i[3:0];
    end
  end

  wire overflow = (lod == 4'd13) || (lod == 4'd14);
  wire [3:0] rshift_amt = (lod > 4'd12) ? (lod - 4'd12) : 4'd0;
  wire [3:0] lshift_amt = (lod < 4'd12) ? (4'd12 - lod) : 4'd0;

  wire [14:0] norm_mant = sum_is_zero ? 15'd0 :
                          overflow    ? (mant_sum >> rshift_amt) :
                                        (mant_sum << lshift_amt);

  wire signed [6:0] lg_exp_s  = $signed({2'b0, lg_exp});
  wire signed [6:0] rsh_s     = $signed({3'b0, rshift_amt});
  wire signed [6:0] lsh_s     = $signed({3'b0, lshift_amt});
  wire signed [6:0] exp_adj_s = sum_is_zero ? 7'sd0 :
                                overflow    ? (lg_exp_s + rsh_s) :
                                              (lg_exp_s - lsh_s);

  wire [9:0] trunc_mant    = norm_mant[11:2];
  wire       guard_bit     = norm_mant[1];
  wire       round_bit     = norm_mant[0];
  wire       extra_sticky  = overflow ? |mant_sum[0] : 1'b0;
  wire       sticky_bit    = sticky | extra_sticky;
  wire       use_sticky    = sticky_bit & ~eff_sub;
  wire       round_up      = guard_bit & (round_bit | use_sticky | trunc_mant[0]);

  wire [10:0] rounded_mant  = {1'b0, trunc_mant} + {10'd0, round_up};
  wire        round_ovf     = rounded_mant[10];
  wire signed [6:0] final_exp_s = round_ovf ? (exp_adj_s + 7'sd1) : exp_adj_s;

  wire [15:0] normal_result = {lg_sign, final_exp_s[4:0], rounded_mant[9:0]};
  wire exp_overflow  = (final_exp_s >= 7'sd31);
  wire [15:0] inf_result  = {lg_sign, 5'd31, 10'd0};
  wire exp_underflow = (final_exp_s <= 7'sd0) && !sum_is_zero;
  wire [15:0] zero_result = {lg_sign, 15'd0};
  wire [15:0] nan_result  = 16'h7E00;

  reg [15:0] result;
  always @(*) begin
    if (a_is_nan || b_is_nan) begin
      result = nan_result;
    end else if (a_is_inf && b_is_inf && eff_sub) begin
      result = nan_result;
    end else if (a_is_inf || b_is_inf) begin
      result = a_is_inf ? {a_sign, 5'd31, 10'd0} : {b_sign, 5'd31, 10'd0};
    end else if (a_is_zero && b_is_zero) begin
      result = {a_sign & b_sign, 15'd0};
    end else if (a_is_zero) begin
      result = b_i;
    end else if (b_is_zero) begin
      result = a_i;
    end else if (sum_is_zero) begin
      result = 16'd0;
    end else if (exp_underflow) begin
      result = zero_result;
    end else if (exp_overflow) begin
      result = inf_result;
    end else begin
      result = normal_result;
    end
  end
  assign sum_o = result;

endmodule
