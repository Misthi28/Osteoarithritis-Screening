"""Model integration contract. Replace only predict_oa_risk() later."""

def predict_oa_risk(patient: dict, gait: dict | None) -> dict:
    """Temporary development result; not a clinical model."""
    factors = {
        "Pain level": patient["pain_score"] / 10 * 0.30,
        "Morning stiffness": min(patient["stiffness_min"] / 60, 1) * 0.15,
        "Mobility difficulty": min(len(patient["difficulties"]) / 4, 1) * 0.15,
        "Age": min(max((patient["age"] - 40) / 40, 0), 1) * 0.15,
        "Previous injury / family history": (
            (patient["past_injury"] == "Yes") + (patient["family_history"] == "Yes")
        ) * 0.05,
    }
    if gait:
        factors["Gait speed"] = min(max((1.2 - gait["gait_speed"]) / 0.6, 0), 1) * 0.10
        factors["Knee motion range"] = min(max((60 - gait["knee_rom"]) / 25, 0), 1) * 0.10
    score = min(sum(factors.values()), 1.0)
    level = "Low" if score < .33 else "Moderate" if score < .60 else "High"
    severity = "None/Mild" if score < .40 else "Moderate" if score < .65 else "Severe"
    return {"risk_score": score, "risk_level": level, "severity": severity,
            "factors": factors, "is_placeholder": True}

