// LayerNorm: y_i = (x_i - mean) / sqrt(var) * gamma + beta
//
// Optimisation vs original:
//   - gamma_buf[128] and beta_buf[128] (2 x 128 x fp16 = 4096 flip-flops) REMOVED.
//   - gamma and beta are now read from weight_store during the NORM pass itself,
//     one element per cycle, interleaved with the normalisation arithmetic.
//     This requires the weight_store to be accessible during NORM (it is, because
//     the weight-store mux in the parent gives us the port whenever we assert
//     w_sel_o / w_addr_o).
//   - FSM collapses S_LOAD_GAMMA + S_LOAD_BETA + S_NORM  →  S_NORM_GAMMA + S_NORM_BETA
//     Two-pass norm: first pass reads gamma and computes (x-mean)*inv_std*gamma,
//     second pass reads beta and adds beta. Alternatively, we do a single pass
//     but gamma must be available 2 cycles early (BRAM pipeline) while beta is
//     available 2 cycles early from the beta base address.
//
//   SINGLE-PASS APPROACH (chosen):
//     We issue gamma addr and beta addr simultaneously by using w_sel_o for gamma
//     (gamma_sel_i) and a local register for beta. But the weight_store has only
//     one data port (w_data_i), so we CANNOT read gamma and beta simultaneously.
//
//   TWO-PASS APPROACH (chosen for correctness):
//     Pass A (S_NORM_G, 130 cycles): read gamma from weight_store, compute
//       partial = (x[i] - mean) * inv_std * gamma[i], store in y_o[i].
//     Pass B (S_NORM_B, 130 cycles): read beta, add beta[i] to y_o[i].
//     Total norm latency: 260 cycles vs original 128+130+130 = 388 cycles — FASTER.
//     gamma_buf and beta_buf eliminated: saves 4096 FF.
//
// FSM: IDLE -> MEAN_ACC(128) -> MEAN_DIV(1) -> VAR_ACC(128) ->
//      VAR_DIV(1) -> INV_SQRT(2) -> NORM_G(130) -> NORM_B(130) -> DONE
// Latency: ~522 cycles (was ~646)

module layernorm #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,

  // Input: DIM x fp16 (flat bus)
  input  wire [DIM*16-1:0] x_i,

  // Weight store interface for gamma/beta
  output reg  [5:0]  w_sel_o,
  output reg  [6:0]  w_addr_o,
  input  wire [7:0]  w_data_i,

  // Which LN instance: tensor_sel for gamma, gamma+1 for beta
  input  wire [5:0]  gamma_sel_i,

  // FP16 dequant scales for gamma and beta
  input  wire [15:0] gamma_scale_i,
  input  wire [15:0] beta_scale_i,

  // Output: DIM x fp16 (flat bus)
  output reg  [DIM*16-1:0] y_o,
  output reg               done_o,
  output reg               busy_o
);

  localparam S_IDLE    = 4'd0;
  localparam S_MEAN_ACC = 4'd1;
  localparam S_MEAN_DIV = 4'd2;
  localparam S_VAR_ACC  = 4'd3;
  localparam S_VAR_DIV  = 4'd4;
  localparam S_INV_SQRT = 4'd5;
  localparam S_NORM_G   = 4'd6;   // read gamma + compute partial y
  localparam S_NORM_B   = 4'd7;   // read beta  + add beta to y

  reg [3:0] state;
  reg [7:0] idx;

  // FP16 accumulators
  reg [15:0] sum_acc;
  reg [15:0] neg_mean;
  reg [15:0] var_acc;
  reg [15:0] inv_std;

  // fp16(1/128)
  localparam [15:0] INV_N = 16'h2000;

  // Current input element
  wire [15:0] x_elem = x_i[idx[6:0]*16 +: 16];

  // BRAM pipeline offset: addr registered T, data valid T+2
  wire [7:0] prev = idx - 8'd1;   // element whose data just arrived
  wire [7:0] prev2 = idx - 8'd2;  // element written to y_o this cycle

  // -----------------------------------------------------------------------
  // Combinational fp16 arithmetic — shared across states
  // -----------------------------------------------------------------------

  // MEAN_ACC
  wire [15:0] mean_add_out;
  fp16_add_comb u_mean_add (.a_i(sum_acc), .b_i(x_elem), .sum_o(mean_add_out));

  // MEAN_DIV: sum * (1/128)
  wire [15:0] mean_div_out;
  fp16_mul_comb u_mean_div (.a_i(sum_acc), .b_i(INV_N), .prod_o(mean_div_out));

  // VAR_ACC: diff = x - mean, sq = diff^2
  wire [15:0] var_diff;
  fp16_add_comb u_var_sub (.a_i(x_elem), .b_i(neg_mean), .sum_o(var_diff));

  wire [15:0] var_sq;
  fp16_mul_comb u_var_sq (.a_i(var_diff), .b_i(var_diff), .prod_o(var_sq));

  wire [15:0] var_add_out;
  fp16_add_comb u_var_add (.a_i(var_acc), .b_i(var_sq), .sum_o(var_add_out));

  // VAR_DIV
  wire [15:0] var_div_out;
  fp16_mul_comb u_var_div (.a_i(var_acc), .b_i(INV_N), .prod_o(var_div_out));

  // fp16_rsqrt
  reg         rsqrt_valid;
  wire        rsqrt_done;
  wire [15:0] rsqrt_result;

  fp16_rsqrt u_rsqrt (
    .clk_i   (clk_i),
    .valid_i (rsqrt_valid),
    .val_i   (var_div_out),
    .valid_o (rsqrt_done),
    .result_o(rsqrt_result)
  );

  // Dequant: int8 from weight_store -> fp16
  wire [15:0] dequant_fp16;
  fp16_from_int8 u_dequant (.val_i(w_data_i), .fp16_o(dequant_fp16));

  // Dequant * gamma_scale  (used in S_NORM_G)
  wire [15:0] dequant_gamma;
  fp16_mul_comb u_deq_gamma (.a_i(dequant_fp16), .b_i(gamma_scale_i), .prod_o(dequant_gamma));

  // Dequant * beta_scale   (used in S_NORM_B)
  wire [15:0] dequant_beta;
  fp16_mul_comb u_deq_beta (.a_i(dequant_fp16), .b_i(beta_scale_i), .prod_o(dequant_beta));

  // NORM_G: partial = (x[i] - mean) * inv_std * gamma[i]
  // We need x_elem for the element whose gamma just arrived (prev2).
  wire [15:0] norm_x_elem = x_i[prev2[6:0]*16 +: 16];

  wire [15:0] norm_diff;
  fp16_add_comb u_norm_sub (.a_i(norm_x_elem), .b_i(neg_mean), .sum_o(norm_diff));

  wire [15:0] norm_scaled;
  fp16_mul_comb u_norm_mul1 (.a_i(norm_diff), .b_i(inv_std), .prod_o(norm_scaled));

  wire [15:0] norm_gamma_val;
  fp16_mul_comb u_norm_mul2 (.a_i(norm_scaled), .b_i(dequant_gamma), .prod_o(norm_gamma_val));

  // NORM_B: y[i] += beta[i]
  wire [15:0] norm_y_elem = y_o[prev2[6:0]*16 +: 16];
  wire [15:0] norm_out;
  fp16_add_comb u_norm_add (.a_i(norm_y_elem), .b_i(dequant_beta), .sum_o(norm_out));

  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      idx         <= 8'd0;
      sum_acc     <= 16'd0;
      var_acc     <= 16'd0;
      neg_mean    <= 16'd0;
      inv_std     <= 16'd0;
      rsqrt_valid <= 1'b0;
      done_o      <= 1'b0;
      busy_o      <= 1'b0;
      w_sel_o     <= 6'd0;
      w_addr_o    <= 7'd0;

    end else begin
      done_o      <= 1'b0;
      rsqrt_valid <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state   <= S_MEAN_ACC;
            idx     <= 8'd0;
            sum_acc <= 16'd0;
            busy_o  <= 1'b1;
          end
        end

        // Pass 1: accumulate sum
        S_MEAN_ACC: begin
          sum_acc <= mean_add_out;
          idx     <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1)
            state <= S_MEAN_DIV;
        end

        // Compute mean, negate
        S_MEAN_DIV: begin
          neg_mean <= {~mean_div_out[15], mean_div_out[14:0]};
          var_acc  <= 16'd0;
          idx      <= 8'd0;
          state    <= S_VAR_ACC;
        end

        // Pass 2: accumulate variance
        S_VAR_ACC: begin
          var_acc <= var_add_out;
          idx     <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1)
            state <= S_VAR_DIV;
        end

        // Compute var, launch rsqrt
        S_VAR_DIV: begin
          rsqrt_valid <= 1'b1;
          idx         <= 8'd0;
          state       <= S_INV_SQRT;
        end

        // Wait for rsqrt (2 cycles)
        S_INV_SQRT: begin
          if (rsqrt_done) begin
            inv_std  <= rsqrt_result;
            state    <= S_NORM_G;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i;
            w_addr_o <= 7'd0;
          end
        end

        // Pass 3a: read gamma from weight_store (2-cycle BRAM latency),
        // compute partial y[i] = (x[i]-mean)*inv_std*gamma[i]
        //
        // idx=0 : issue addr 0
        // idx=1 : issue addr 1, addr 0 latched in BRAM
        // idx=2 : issue addr 2, data[0] valid -> compute partial[0] -> write y_o[0]
        // ...
        // idx=DIM+1: last write y_o[DIM-1]
        S_NORM_G: begin
          // Advance address: issue addr idx+1 so data arrives at idx+2
          if (idx < DIM[7:0] - 8'd1)
            w_addr_o <= idx[6:0] + 7'd1;

          // Write partial result when data is valid (2 cycles after addr)
          if (idx >= 8'd2) begin
            y_o[prev2[6:0]*16 +: 16] <= norm_gamma_val;
          end

          idx <= idx + 8'd1;

          if (idx == DIM[7:0] + 8'd1) begin
            // All partials written; now add beta
            state    <= S_NORM_B;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i + 6'd1;  // beta tensor
            w_addr_o <= 7'd0;
          end
        end

        // Pass 3b: read beta, add to y_o[i]
        S_NORM_B: begin
          if (idx < DIM[7:0] - 8'd1)
            w_addr_o <= idx[6:0] + 7'd1;

          if (idx >= 8'd2) begin
            y_o[prev2[6:0]*16 +: 16] <= norm_out;
          end

          idx <= idx + 8'd1;

          if (idx == DIM[7:0] + 8'd1) begin
            state  <= S_IDLE;
            done_o <= 1'b1;
            busy_o <= 1'b0;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
