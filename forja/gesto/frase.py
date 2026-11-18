"""De la frase a la intención, con plantillas de verbo + objeto (sin modelos ni servicios externos).

Una plantilla es una expresión regular sobre la frase normalizada (minúsculas, sin artículos ni signos). Si ninguna
encaja, se dice cuáles hay; no se adivina.
"""
import re
import unicodedata

ARTICULOS = r"\b(the|a|an|that|this|el|la|los|las|un|una|esa|ese|esta|este)\b"
OBJETOS = {
    "box": "caja", "crate": "caja", "package": "caja", "parcel": "caja", "caja": "caja", "paquete": "caja",
}
OBJ = "(?P<obj>" + "|".join(OBJETOS) + ")"

# (gesto, expresión). El orden importa: lo más concreto primero.
PLANTILLAS = [
    ("recoger", rf"^(pick|picks|picking|lift|lifts|lifting|grab|grabs)( up)? {OBJ}( up)?( (from|off) (ground|floor))?$"),
    ("recoger", rf"^(pick|picks|picking|lift|lifts) {OBJ} up( (from|off) (ground|floor))?$"),
    ("recoger", rf"^(recoge|recoger|levanta|levantar|coge|coger|agarra|agarrar) {OBJ}( del (suelo|piso))?$"),
    ("dejar", rf"^(put|puts|putting|set|sets|place|places|lower|lowers) {OBJ} (down|on (ground|floor))$"),
    ("dejar", rf"^(put|puts|set|sets|place|places) down {OBJ}$"),
    ("dejar", rf"^(deja|dejar|baja|bajar|pon|poner|suelta|soltar) {OBJ}( en el (suelo|piso))?$"),
]


def normalizar(frase):
    s = unicodedata.normalize("NFKD", frase.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(ARTICULOS, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def interpretar(frase):
    """→ {"gesto": "recoger"|"dejar", "objeto": "caja", "frase": …, "normalizada": …, "plantilla": i}."""
    n = normalizar(frase)
    for i, (gesto, rx) in enumerate(PLANTILLAS):
        m = re.match(rx, n)
        if m:
            return {"gesto": gesto, "objeto": OBJETOS[m.group("obj")], "frase": frase, "normalizada": n, "plantilla": i}
    ejemplos = "«pick up box from ground», «put box down», «recoge la caja del suelo», «deja la caja»"
    raise SystemExit(f"no entiendo «{frase}» (normalizada: «{n}»). Plantillas de hoy: {ejemplos}.")
