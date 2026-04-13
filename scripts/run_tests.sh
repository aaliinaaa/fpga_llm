# #!/usr/bin/env bash
# # Run testbenches using ModelSim/QuestaSim for a Quartus-based project.
# set -euo pipefail

# # Project layout
# PROJ="C:/Users/alina/Desktop/red_eyes_is_all_you_need_cyclonev2/red_eyes_is_all_you_need_cyclonev"
# QUARTUS_PROJ_DIR="$PROJ/quartus_project"

# # Script path helpers
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# REPO_ROOT="$SCRIPT_DIR/.."

# sim_fails=0
# val_fails=0
# sim_total=0
# val_total=0

# # Build list of testbenches to run
# if [ $# -gt 0 ]; then
#   tb_files=""
#   for name in "$@"; do
#     tb_files="$tb_files $PROJ/tb/tb_${name}.v"
#   done
# else
#   tb_files=$(ls "$PROJ/tb/tb_"*.v 2>/dev/null || true)
# fi

# echo "Running simulations"
# echo ""

# # Simple validator map (kept compatible with existing repo structure)
# declare -A VAL_MAP
# VAL_MAP=(
#   [tb_fp16]=validate_fp16.py
#   [tb_attention]=validate_attention.py
#   [tb_attention_stress]="validate_attention.py logs/tb_attention_stress.log"
#   [tb_embedding]=validate_embedding.py
#   [tb_gelu]=validate_gelu.py
#   [tb_kv_cache]=validate_kv_cache.py
#   [tb_layernorm]=validate_layernorm.py
#   [tb_softmax]=validate_softmax.py
#   [tb_transformer_layer]=validate_transformer_layer.py
#   [tb_transformer_layer_stress]="validate_transformer_layer.py logs/tb_transformer_layer_stress.log"
#   [tb_transformer_top]=validate_transformer_top.py
#   [tb_transformer_top_stress]="validate_transformer_top.py logs/tb_transformer_top_stress.log"
#   [tb_weight_store]=validate_weights.py
# )

# for tb_path in $tb_files; do
#   tb_file=$(basename "$tb_path" .v)
#   sim_total=$((sim_total + 1))
#   echo ">>> $tb_file"

#   # Run ModelSim/QuestaSim flow: vlog then vsim
#   output=$(bash -lc "
#     rm -rf /tmp/xsim_$tb_file && mkdir -p /tmp/xsim_$tb_file && cd /tmp/xsim_$tb_file
#     vlog -work work $PROJ/rtl/*.v $PROJ/tb/${tb_file}.v 2>&1 || true
#     vsim -c work.${tb_file} -do 'run -all' 2>&1
#   ") || true

#   if echo "$output" | grep -qiE "^(ERROR|FAIL|TIMEOUT)"; then
#     echo "  FAIL"
#     echo "$output" | grep -iE "^(ERROR|FAIL|TIMEOUT)"
#     sim_fails=$((sim_fails + 1))
#   else
#     echo "  PASS"
#   fi
# done

# echo ""
# echo "Running validation scripts"
# echo ""

# for tb_path in $tb_files; do
#   tb_file=$(basename "$tb_path" .v)
#   # Reuse same mapping as original project for validators
#   val_entry="${VAL_MAP[$tb_file]:-}"
#   if [ -z "$val_entry" ]; then
#     echo ">>> $tb_file: no validator, skipping"
#     continue
#   fi
#   script="${val_entry%% *}"
#   args="${val_entry#"$script"}"
#   script_path="$REPO_ROOT/scripts/tests/$script"
#   if [ ! -f "$script_path" ]; then
#     echo ">>> $script: not found, skipping"
#     continue
#   fi
#   val_total=$((val_total + 1))
#   echo ">>> $script${args}"
#   if output=$(python3 "$script_path" $args 2>&1); then
#     echo "  $(echo "$output" | tail -1)"
#   else
#     echo "  FAIL"
#     echo "$output" | tail -3
#     val_fails=$((val_fails + 1))
#   fi
# done

# echo ""
# total_fails=$((sim_fails + val_fails))
# if [ $total_fails -eq 0 ]; then
#   echo "=== All $sim_total sims + $val_total validations passed ==="
# else
#   echo "=== $total_fails failures ($sim_fails sim, $val_fails val) ==="
#   exit 1
# fi

#!/usr/bin/env bash
# Run testbenches using ModelSim for a Quartus-based Cyclone V project.

PROJ="C:/Users/alina/Desktop/red_eyes_is_all_you_need_cyclonev2/red_eyes_is_all_you_need_cyclonev"
LOGS="$PROJ/logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."

sim_fails=0
val_fails=0
sim_total=0
val_total=0

# Validator map: testbench name -> "script.py [optional_log_override]"
declare -A VAL_MAP
VAL_MAP=(
  [tb_fp16]=validate_fp16.py
  [tb_attention]=validate_attention.py
  [tb_attention_stress]="validate_attention.py $LOGS/tb_attention_stress.log"
  [tb_embedding]=validate_embedding.py
  [tb_gelu]=validate_gelu.py
  [tb_kv_cache]=validate_kv_cache.py
  [tb_layernorm]=validate_layernorm.py
  [tb_softmax]=validate_softmax.py
  [tb_transformer_layer]=validate_transformer_layer.py
  [tb_transformer_layer_stress]="validate_transformer_layer.py $LOGS/tb_transformer_layer_stress.log"
  [tb_transformer_top]=validate_transformer_top.py
  [tb_transformer_top_stress]="validate_transformer_top.py $LOGS/tb_transformer_top_stress.log"
  [tb_weight_store]=validate_weights.py
)

# Build list of testbenches
if [ $# -gt 0 ]; then
  tb_files=""
  for name in "$@"; do
    tb_files="$tb_files $PROJ/tb/tb_${name}.v"
  done
else
  tb_files=$(ls "$PROJ/tb/tb_"*.v 2>/dev/null || true)
fi

if [ -z "$tb_files" ]; then
  echo "No testbench files found in $PROJ/tb/"
  exit 1
fi

# Make sure logs directory exists
mkdir -p "$LOGS"

# ── Compile all RTL + testbenches once ──────────────────────────────────────
echo "=== Compiling RTL + testbenches ==="

# Build tb file list for vlog
tb_list=""
for tb_path in $tb_files; do
  tb_list="$tb_list $tb_path"
done

compile_out=$(vlog -work work \
  +incdir+"$PROJ/rtl" \
  "$PROJ/rtl/"*.v \
  $tb_list 2>&1)
compile_status=$?

echo "$compile_out" | grep -E "^\*\* (Error|Warning)" || true

if [ $compile_status -ne 0 ] || echo "$compile_out" | grep -q "^\*\* Error"; then
  echo "COMPILE FAILED — stopping"
  echo "$compile_out"
  exit 1
fi
echo "Compile OK"
echo ""

# ── Run simulations ──────────────────────────────────────────────────────────
echo "=== Running simulations ==="
echo ""

for tb_path in $tb_files; do
  tb_file=$(basename "$tb_path" .v)
  sim_total=$((sim_total + 1))
  echo ">>> Simulating: $tb_file"

  # Run vsim from PROJ dir so relative paths in testbenches resolve correctly
  sim_out=$(cd "$PROJ" && vsim -c "work.$tb_file" \
    -do "run -all; quit" 2>&1) || true

  # Check for real vsim errors (not testbench $display output)
  # We look for vsim-level errors, not words inside simulation output
  if echo "$sim_out" | grep -qE "^\*\* Error|^# \*\* Error|TIMEOUT"; then
    echo "  [FAIL] vsim error or timeout"
    echo "$sim_out" | grep -E "^\*\* Error|^# \*\* Error|TIMEOUT"
    sim_fails=$((sim_fails + 1))
  else
    echo "  [PASS] simulation finished"
  fi

  echo ""
done

# ── Run validation scripts ───────────────────────────────────────────────────
echo "=== Running validation scripts ==="
echo ""

for tb_path in $tb_files; do
  tb_file=$(basename "$tb_path" .v)
  val_entry="${VAL_MAP[$tb_file]:-}"

  if [ -z "$val_entry" ]; then
    echo ">>> $tb_file: no validator, skipping"
    continue
  fi

  script="${val_entry%% *}"
  args="${val_entry#"$script"}"
  script_path="$REPO_ROOT/scripts/tests/$script"

  if [ ! -f "$script_path" ]; then
    echo ">>> $script: script not found at $script_path, skipping"
    continue
  fi

  val_total=$((val_total + 1))
  echo ">>> Validating: $script${args:+ $args}"

  if val_out=$(cd "$PROJ" && python "$script_path" $args 2>&1); then
    last=$(echo "$val_out" | tail -1)
    echo "  [PASS] $last"
  else
    echo "  [FAIL]"
    echo "$val_out" | tail -5 | sed 's/^/    /'
    val_fails=$((val_fails + 1))
  fi

  echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
total_fails=$((sim_fails + val_fails))
echo "════════════════════════════════════════"
if [ $total_fails -eq 0 ]; then
  echo "ALL PASSED — $sim_total sims, $val_total validations"
else
  echo "FAILURES: $sim_fails sim + $val_fails validation = $total_fails total"
  echo "          (out of $sim_total sims, $val_total validations)"
  exit 1
fi