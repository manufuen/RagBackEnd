import re
import unicodedata


TOPIC_KEYWORDS = {
    "finanzas": [
        "factura", "balance", "ingresos", "gastos", "presupuesto",
        "financiero", "contabilidad", "beneficio", "perdida",
        "inversion", "iva", "impuesto", "banco", "capital"
    ],
    "legal": [
        "contrato", "clausula", "ley", "legal", "juridico",
        "arrendamiento", "demanda", "sentencia", "obligacion",
        "derecho", "normativa", "acuerdo", "firma"
    ],
    "tecnologia": [
        "api", "software", "servidor", "base de datos", "backend",
        "frontend", "codigo", "python", "java", "cloud", "modelo",
        "inteligencia artificial", "machine learning", "red neuronal"
    ],
    "salud": [
        "paciente", "medico", "salud", "diagnostico", "tratamiento",
        "hospital", "enfermedad", "clinico", "farmaco", "medicamento"
    ],
    "educacion": [
        "alumno", "profesor", "curso", "asignatura", "examen",
        "universidad", "colegio", "aprendizaje", "tema", "evaluacion"
    ],
    "rrhh": [
        "empleado", "nomina", "vacaciones", "contratacion",
        "recursos humanos", "entrevista", "puesto", "salario",
        "candidato", "trabajador"
    ],
}


ALLOWED_TOPICS = [
    "finanzas",
    "legal",
    "tecnologia",
    "salud",
    "educacion",
    "rrhh",
    "otros",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text


def classify_document(text: str) -> str:
    normalized_text = normalize_text(text[:5000])

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            matches = re.findall(rf"\b{re.escape(normalized_keyword)}\b", normalized_text)
            score += len(matches)

        scores[topic] = score

    best_topic = max(scores, key=scores.get)

    if scores[best_topic] == 0:
        return "otros"

    return best_topic


def classify_question(question: str) -> str:
    return classify_document(question)