def predict_market_reaction(risk_score):
    if risk_score > 70:
        return "DOWN"
    elif risk_score < 40:
        return "UP"
    return "NEUTRAL"
