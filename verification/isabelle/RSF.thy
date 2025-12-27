theory RSF
  imports Complex_Main Tensor
begin

record ('dim) rsf_layer_weights =
  s_weight :: "nat list"
  t_weight :: "nat list"
  s_bias :: "nat list"
  t_bias :: "nat list"

definition rsf_layer_init :: "nat \<Rightarrow> ('dim) rsf_layer_weights" where
  "rsf_layer_init dim \<equiv> \<lparr>
    s_weight = replicate (dim * dim) 0,
    t_weight = replicate (dim * dim) 0,
    s_bias = replicate dim 0,
    t_bias = replicate dim 0 \<rparr>"

record ('dim) rsf_layer_gradients =
  s_weight_grad :: "nat list"
  t_weight_grad :: "nat list"
  s_bias_grad :: "nat list"
  t_bias_grad :: "nat list"

definition rsf_gradients_init :: "nat \<Rightarrow> ('dim) rsf_layer_gradients" where
  "rsf_gradients_init dim \<equiv> \<lparr>
    s_weight_grad = replicate (dim * dim) 0,
    t_weight_grad = replicate (dim * dim) 0,
    s_bias_grad = replicate dim 0,
    t_bias_grad = replicate dim 0 \<rparr>"

record ('dim) rsf_layer_state =
  weights :: "('dim) rsf_layer_weights"
  gradients :: "('dim) rsf_layer_gradients"

definition rsf_layer_state_init :: "nat \<Rightarrow> ('dim) rsf_layer_state" where
  "rsf_layer_state_init dim \<equiv> \<lparr>
    weights = rsf_layer_init dim,
    gradients = rsf_gradients_init dim \<rparr>"

fun matrix_vector_mul :: "nat \<Rightarrow> nat \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> nat list" where
  "matrix_vector_mul rows cols mat vec =
    (if length mat = rows * cols \<and> length vec = cols
     then map (\<lambda>i. sum_list (map (\<lambda>j. mat ! (i * cols + j) * vec ! j) [0..<cols])) [0..<rows]
     else [])"

fun vector_add :: "nat list \<Rightarrow> nat list \<Rightarrow> nat list" where
  "vector_add v1 v2 = (if length v1 = length v2 then map (\<lambda>(x, y). x + y) (zip v1 v2) else [])"

fun vector_mul :: "nat list \<Rightarrow> nat list \<Rightarrow> nat list" where
  "vector_mul v1 v2 = (if length v1 = length v2 then map (\<lambda>(x, y). x * y) (zip v1 v2) else [])"

fun clamp_nat :: "nat \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> nat" where
  "clamp_nat x min_val max_val =
    (if x < min_val then min_val
     else if max_val < x then max_val
     else x)"

fun vector_exp :: "nat list \<Rightarrow> nat list" where
  "vector_exp vec = map (\<lambda>_. 2) vec"

fun rsf_forward_scatter :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_forward_scatter dim weights x1 x2 =
    (let s_out = matrix_vector_mul dim dim (s_weight weights) x2;
         s_bias_added = vector_add s_out (s_bias weights);
         s_clamped = map (\<lambda>x. clamp_nat x 0 10) s_bias_added;
         s_exp = vector_exp s_clamped;
         y1 = vector_mul x1 s_exp
     in (y1, x2))"

fun rsf_forward_transform :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_forward_transform dim weights x1 x2 =
    (let t_out = matrix_vector_mul dim dim (t_weight weights) x1;
         t_bias_added = vector_add t_out (t_bias weights);
         y2 = vector_add x2 t_bias_added
     in (x1, y2))"

definition rsf_forward :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_forward dim weights x1 x2 \<equiv>
    (let (y1_scatter, y2_scatter) = rsf_forward_scatter dim weights x1 x2
     in rsf_forward_transform dim weights y1_scatter y2_scatter)"

fun rsf_inverse_transform :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_inverse_transform dim weights y1 y2 =
    (let t_out = matrix_vector_mul dim dim (t_weight weights) y1;
         t_bias_added = vector_add t_out (t_bias weights);
         x2 = map (\<lambda>(a, b). if a \<ge> b then a - b else 0) (zip y2 t_bias_added)
     in (y1, x2))"

fun rsf_inverse_scatter :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_inverse_scatter dim weights y1 y2 =
    (let s_out = matrix_vector_mul dim dim (s_weight weights) y2;
         s_bias_added = vector_add s_out (s_bias weights);
         s_clamped = map (\<lambda>x. clamp_nat x 0 10) s_bias_added;
         s_exp = vector_exp s_clamped;
         x1 = map (\<lambda>(a, b). a div (b + 1)) (zip y1 s_exp)
     in (x1, y2))"

definition rsf_inverse :: "nat \<Rightarrow> ('dim) rsf_layer_weights \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_inverse dim weights y1 y2 \<equiv>
    (let (x1_transform, x2_transform) = rsf_inverse_transform dim weights y1 y2
     in rsf_inverse_scatter dim weights x1_transform x2_transform)"

record ('dim, 'num_layers) rsf_network =
  layers :: "('dim) rsf_layer_state list"

definition rsf_network_init :: "nat \<Rightarrow> nat \<Rightarrow> ('dim, 'num_layers) rsf_network" where
  "rsf_network_init dim num_layers \<equiv> \<lparr>
    layers = replicate num_layers (rsf_layer_state_init dim) \<rparr>"

fun rsf_network_forward :: "nat \<Rightarrow> ('dim, 'num_layers) rsf_network \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_network_forward dim net x1 x2 =
    (case layers net of
       [] \<Rightarrow> (x1, x2)
     | layer # rest \<Rightarrow>
         let (y1, y2) = rsf_forward dim (weights layer) x1 x2;
             rest_net = \<lparr> layers = rest \<rparr>
         in rsf_network_forward dim rest_net y1 y2)"

fun rsf_network_inverse :: "nat \<Rightarrow> ('dim, 'num_layers) rsf_network \<Rightarrow> nat list \<Rightarrow> nat list \<Rightarrow> (nat list \<times> nat list)" where
  "rsf_network_inverse dim net y1 y2 =
    (case rev (layers net) of
       [] \<Rightarrow> (y1, y2)
     | last_layer # rest_rev \<Rightarrow>
         let init_net = \<lparr> layers = rev rest_rev \<rparr>;
             (x1_partial, x2_partial) = rsf_network_inverse dim init_net y1 y2
         in rsf_inverse dim (weights last_layer) x1_partial x2_partial)"

theorem rsf_layer_forward_shape:
  assumes "length x1 = dim" "length x2 = dim"
  shows "let (y1, y2) = rsf_forward dim weights x1 x2
         in length y1 = dim \<and> length y2 = dim"
  using assms by (simp add: rsf_forward_def rsf_forward_scatter.simps rsf_forward_transform.simps)

theorem rsf_layer_inverse_shape:
  assumes "length y1 = dim" "length y2 = dim"
  shows "let (x1, x2) = rsf_inverse dim weights y1 y2
         in length x1 = dim \<and> length x2 = dim"
  using assms by (simp add: rsf_inverse_def rsf_inverse_transform.simps rsf_inverse_scatter.simps)

theorem rsf_network_forward_shape:
  assumes "length x1 = dim" "length x2 = dim"
  shows "let (y1, y2) = rsf_network_forward dim net x1 x2
         in length y1 = dim \<and> length y2 = dim"
  using assms by (induction "layers net" arbitrary: x1 x2) auto

theorem rsf_network_inverse_shape:
  assumes "length y1 = dim" "length y2 = dim"
  shows "let (x1, x2) = rsf_network_inverse dim net y1 y2
         in length x1 = dim \<and> length x2 = dim"
  using assms by (induction "layers net" arbitrary: y1 y2) auto

definition zero_gradients :: "('dim) rsf_layer_gradients \<Rightarrow> ('dim) rsf_layer_gradients" where
  "zero_gradients grads \<equiv> \<lparr>
    s_weight_grad = replicate (length (s_weight_grad grads)) 0,
    t_weight_grad = replicate (length (t_weight_grad grads)) 0,
    s_bias_grad = replicate (length (s_bias_grad grads)) 0,
    t_bias_grad = replicate (length (t_bias_grad grads)) 0 \<rparr>"

theorem zero_gradients_all_zero:
  "let zeroed = zero_gradients grads
   in (\<forall>i < length (s_weight_grad zeroed). s_weight_grad zeroed ! i = 0) \<and>
      (\<forall>i < length (t_weight_grad zeroed). t_weight_grad zeroed ! i = 0)"
  by (simp add: zero_gradients_def)

end
