import math

import pandas as pd

CONVERSION_FACTOR = 173.7178

class Glicko2Player:
    def __init__(self, rating: float = 1500.0, rd: float = 350.0, volatility: float = 0.06, tau: float = 0.5):
        self.rating = rating
        self.rd = rd
        self.volatility = volatility
        self.tau = tau

    @property
    def mu(self) -> float:
        return (self.rating - 1500.0) / CONVERSION_FACTOR

    @property
    def phi(self) -> float:
        return self.rd / CONVERSION_FACTOR

    def set_glicko2_values(self, mu: float, phi: float, volatility: float | None = None):
        self.rating = mu * CONVERSION_FACTOR + 1500.0
        self.rd = phi * CONVERSION_FACTOR
        if volatility is not None:
            self.volatility = volatility

def g_function(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

def glicko2_expected_score(player_a: Glicko2Player, player_b: Glicko2Player) -> float:
    g_phi = g_function(player_b.phi)
    return 1.0 / (1.0 + math.exp(-g_phi * (player_a.mu - player_b.mu)))

def _update_player_glicko2(player: Glicko2Player, opponent: Glicko2Player, score: float) -> tuple[float, float, float]:
    """
    Updates a single player's (mu, phi, volatility) against an opponent for a single match outcome score (1.0 or 0.0).
    Follows Glickman's Glicko-2 specification.
    """
    mu = player.mu
    phi = player.phi
    sigma = player.volatility
    tau = player.tau

    mu_j = opponent.mu
    phi_j = opponent.phi

    g_j = g_function(phi_j)
    e_j = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))

    # Estimated variance v
    v = 1.0 / ((g_j ** 2) * e_j * (1.0 - e_j))

    # Quantity delta
    delta = v * g_j * (score - e_j)

    # Volatility update iteration
    a = math.log(sigma ** 2)

    def f(x: float) -> float:
        ex = math.exp(x)
        d_sq = delta ** 2
        phi_sq = phi ** 2
        denom = phi_sq + v + ex
        term1 = (ex * (d_sq - phi_sq - v - ex)) / (2.0 * (denom ** 2))
        term2 = (x - a) / (tau ** 2)
        return term1 - term2

    # Illinois algorithm for root finding
    eps = 1e-6
    A = a
    if delta ** 2 > (phi ** 2 + v):
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA = f(A)
    fB = f(B)

    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA = fA / 2.0
        B = C
        fB = fC

    new_sigma = math.exp(A / 2.0)

    # Pre-rating period phi star
    phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)

    # New phi and mu
    new_phi = 1.0 / math.sqrt((1.0 / (phi_star ** 2)) + (1.0 / v))
    new_mu = mu + (new_phi ** 2) * g_j * (score - e_j)

    return new_mu, new_phi, new_sigma

def update_glicko2_pair(p1: Glicko2Player, p2: Glicko2Player, score_p1: float) -> tuple[Glicko2Player, Glicko2Player]:
    score_p2 = 1.0 - score_p1
    
    new_mu1, new_phi1, new_sig1 = _update_player_glicko2(p1, p2, score_p1)
    new_mu2, new_phi2, new_sig2 = _update_player_glicko2(p2, p1, score_p2)

    p1_out = Glicko2Player(tau=p1.tau)
    p1_out.set_glicko2_values(new_mu1, new_phi1, new_sig1)

    p2_out = Glicko2Player(tau=p2.tau)
    p2_out.set_glicko2_values(new_mu2, new_phi2, new_sig2)

    return p1_out, p2_out

# Add method to Glicko2Player class
Glicko2Player.update_against = lambda self, opponent, score: update_glicko2_pair(self, opponent, score)

def compute_glicko2_ratings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, Glicko2Player]]:
    """
    Computes Glicko-2 ratings chronologically across all matches.
    """
    players: dict[int, Glicko2Player] = {}

    blue_pre_glicko = []
    white_pre_glicko = []
    blue_post_glicko = []
    white_post_glicko = []
    blue_rds = []
    white_rds = []

    for row in df.itertuples():
        b_id = row.athlete_blue_id
        w_id = row.athlete_white_id

        p_blue = players.get(b_id, Glicko2Player())
        p_white = players.get(w_id, Glicko2Player())

        blue_pre_glicko.append(p_blue.rating)
        white_pre_glicko.append(p_white.rating)

        score_blue = 1.0 if row.winner_color == 'blue' else 0.0

        p_blue_new, p_white_new = update_glicko2_pair(p_blue, p_white, score_blue)

        blue_post_glicko.append(p_blue_new.rating)
        white_post_glicko.append(p_white_new.rating)
        blue_rds.append(p_blue_new.rd)
        white_rds.append(p_white_new.rd)

        players[b_id] = p_blue_new
        players[w_id] = p_white_new

    df_res = df.copy()
    df_res['blue_pre_glicko'] = blue_pre_glicko
    df_res['white_pre_glicko'] = white_pre_glicko
    df_res['blue_post_glicko'] = blue_post_glicko
    df_res['white_post_glicko'] = white_post_glicko
    df_res['blue_rd'] = blue_rds
    df_res['white_rd'] = white_rds

    return df_res, players
