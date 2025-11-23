def build_text(rec):
    if rec.get("source") == "climat":
        return f"{rec['city']} | {rec['month']} | Tmin: {rec['temp_min']} | Tmax: {rec['temp_max']} | Pluie: {rec['precipitation']}"
    elif rec.get("source") == "bulletin":
        return f"[Page {rec.get('page')}] {rec.get('text', '')[:500]}"
    return str(rec)

def build_meta(rec):
    return {k: v for k, v in rec.items() if k not in ["text", "id"]}