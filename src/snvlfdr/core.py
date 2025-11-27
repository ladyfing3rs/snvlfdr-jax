import jax
import jax.numpy as jnp

# Enable 64-bit precision for scientific calculation stability
jax.config.update("jax_enable_x64", True)

@jax.jit
def multinomial_logpmf(x, p):
    """
    Vectorized multinomial log-probability mass function.
    x: Observed counts [N, 4] (Ref, Alt, Other1, Other2)
    p: Probabilities [N, 4]
    """
    # xlogy handles 0 * log(0) = 0 correctly
    # The multinomial coefficient is constant w.r.t hypotheses, so we can omit it
    # for Bayes Factor calculations, but including it doesn't hurt.
    # Here we calculate the proportional log-likelihood: sum(x * log(p))
    return jax.scipy.special.xlogy(x, p).sum(axis=-1)

@jax.jit
def calculate_f0(counts, error_rates):
    """
    Calculate Likelihood under Null Hypothesis (H0).
    H0: All reads are errors.
    P(Ref) = 1-e
    P(Alt) = e/3
    """
    # error_rates: [N] -> [N, 1]
    e = error_rates[:, None]
    
    # Probabilities under H0: [1-e, e/3, e/3, e/3]
    p0 = jnp.hstack([
        1 - e,
        e / 3.0,
        e / 3.0,
        e / 3.0
    ])
    
    # Log-likelihood + exponentiate
    return jnp.exp(multinomial_logpmf(counts, p0))

@jax.jit
def calculate_f1_batch(counts, error_rates, thetas):
    """
    Calculate Likelihood under Alternative Hypothesis (H1).
    H1: Mixture of true variants (theta) and errors.
    
    counts: [N, 4]
    error_rates: [N]
    thetas: [M] (Sampled allele frequencies)
    """
    # Reshape for broadcasting
    # Counts: [N, 1, 4]
    C = counts[:, None, :] 
    
    # Error: [N, 1, 1]
    E = error_rates[:, None, None]
    
    # Theta: [1, M, 1]
    T = thetas[None, :, None]
    
    # Probabilities under H1 for each theta:
    # Ref: theta * (e/3) + (1-theta) * (1-e)
    # Alt: theta * (1-e) + (1-theta) * (e/3)
    # Others: e/3
    
    p_ref = T * (E/3.0) + (1-T) * (1-E)
    p_alt = T * (1-E) + (1-T) * (E/3.0)
    p_others = E / 3.0
    
    # Construct probability tensor [N, M, 4]
    # We broadcast p_others to match M dimension
    p_others_bd = jnp.tile(p_others, (1, thetas.shape[0], 1))
    
    # Stack: Ref (idx 0), Alt (idx 1), Other1 (idx 2), Other2 (idx 3)
    probs = jnp.concatenate([p_ref, p_alt, p_others_bd, p_others_bd], axis=2)
    
    # Calculate log likelihood for every site against every theta
    # Input X must be broadcast to [N, M, 4]
    log_L = jax.scipy.special.xlogy(C, probs).sum(axis=2)
    
    # Average over all sampled thetas (Monte Carlo integration)
    # mean(likelihood) -> mean(exp(log_likelihood))
    return jnp.mean(jnp.exp(log_L), axis=1)

@jax.jit
def calculate_lfdr(pi0, f0, f1):
    """
    Bayes Rule for LFDR.
    LFDR = P(H0 | Data)
    """
    numerator = pi0 * f0
    denominator = numerator + (1 - pi0) * f1
    
    # Avoid division by zero
    denominator = jnp.where(denominator == 0, 1e-300, denominator)
    return numerator / denominator
