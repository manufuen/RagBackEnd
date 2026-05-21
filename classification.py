import re
import unicodedata
from pathlib import Path


TOPIC_KEYWORDS = {
    "astronomia_universo": [
        "universo", "big bang", "cosmos", "galaxia", "galaxias", "estrella",
        "estrellas", "planeta", "planetas", "sistema solar", "agujero negro",
        "nebulosa", "astronomia", "cosmologia", "espacio", "tiempo",
        "telescopio", "satellite", "satelite", "quarks", "neutrinos",
        "electrones", "hidrogeno", "helio", "materia oscura", "energia oscura"
    ],
    "ciencia_general": [
        "ciencia", "cientifico", "investigacion", "experimento", "hipotesis",
        "teoria", "metodo cientifico", "laboratorio", "observacion", "evidencia"
    ],
    "fisica": [
        "fisica", "fuerza", "energia", "masa", "movimiento", "velocidad",
        "aceleracion", "gravedad", "relatividad", "mecanica", "cuantica"
    ],
    "quimica": [
        "quimica", "molecula", "atomo", "reaccion", "compuesto", "elemento",
        "tabla periodica", "enlace", "acido", "base", "solucion"
    ],
    "biologia": [
        "biologia", "celula", "adn", "gen", "organismo", "evolucion",
        "ecosistema", "proteina", "enzima", "bacteria", "virus"
    ],
    "medicina_salud": [
        "medico", "paciente", "salud", "diagnostico", "tratamiento",
        "hospital", "enfermedad", "farmaco", "medicamento", "clinico"
    ],
    "psicologia": [
        "psicologia", "conducta", "emocion", "mente", "ansiedad",
        "depresion", "terapia", "cognitivo", "personalidad", "aprendizaje"
    ],
    "educacion": [
        "educacion", "alumno", "profesor", "curso", "asignatura", "examen",
        "universidad", "colegio", "aprendizaje", "evaluacion", "docente"
    ],
    "matematicas": [
        "matematicas", "algebra", "calculo", "geometria", "integral",
        "derivada", "ecuacion", "matriz", "probabilidad", "estadistica"
    ],
    "tecnologia": [
        "tecnologia", "software", "hardware", "servidor", "cloud",
        "digital", "sistema", "plataforma", "aplicacion", "dispositivo"
    ],
    "programacion": [
        "programacion", "codigo", "python", "java", "javascript", "api",
        "backend", "frontend", "funcion", "clase", "framework", "bug"
    ],
    "inteligencia_artificial": [
        "inteligencia artificial", "ia", "machine learning", "deep learning",
        "modelo", "red neuronal", "llm", "rag", "embedding", "chatbot"
    ],
    "ciberseguridad": [
        "ciberseguridad", "vulnerabilidad", "malware", "phishing",
        "firewall", "cifrado", "ataque", "amenaza", "seguridad informatica"
    ],
    "redes_sistemas": [
        "red", "redes", "ip", "dns", "servidor", "router", "switch",
        "linux", "windows server", "protocolo", "tcp", "udp"
    ],
    "legal": [
        "contrato", "clausula", "ley", "legal", "juridico", "demanda",
        "sentencia", "obligacion", "derecho", "normativa", "firma"
    ],
    "finanzas": [
        "finanzas", "factura", "balance", "ingresos", "gastos",
        "presupuesto", "inversion", "capital", "rentabilidad", "banco"
    ],
    "contabilidad": [
        "contabilidad", "asiento contable", "cuenta", "debe", "haber",
        "activo", "pasivo", "patrimonio", "amortizacion", "libro diario"
    ],
    "economia": [
        "economia", "inflacion", "pib", "mercado", "oferta", "demanda",
        "macroeconomia", "microeconomia", "recesion", "crecimiento"
    ],
    "rrhh": [
        "recursos humanos", "empleado", "nomina", "vacaciones",
        "contratacion", "entrevista", "puesto", "salario", "trabajador"
    ],
    "marketing": [
        "marketing", "campaña", "marca", "publicidad", "seo", "redes sociales",
        "audiencia", "conversion", "posicionamiento", "cliente"
    ],
    "ventas": [
        "ventas", "comercial", "cliente", "lead", "crm", "pipeline",
        "negociacion", "propuesta", "oferta", "facturacion"
    ],
    "empresa_negocio": [
        "empresa", "negocio", "estrategia", "modelo de negocio", "kpi",
        "objetivo", "operaciones", "direccion", "gestion", "organizacion"
    ],
    "proyectos": [
        "proyecto", "planificacion", "tarea", "hito", "cronograma",
        "scrum", "kanban", "riesgo", "entregable", "sprint"
    ],
    "logistica": [
        "logistica", "almacen", "inventario", "suministro", "stock",
        "distribucion", "transporte", "pedido", "cadena de suministro"
    ],
    "industria_ingenieria": [
        "ingenieria", "industrial", "maquina", "proceso", "fabricacion",
        "produccion", "mantenimiento", "planta", "automatizacion"
    ],
    "arquitectura_construccion": [
        "arquitectura", "construccion", "obra", "edificio", "plano",
        "estructura", "cemento", "hormigon", "vivienda", "licencia"
    ],
    "energia_medioambiente": [
        "energia", "medioambiente", "sostenibilidad", "renovable",
        "solar", "eolica", "contaminacion", "emisiones", "clima", "co2"
    ],
    "agricultura": [
        "agricultura", "cultivo", "cosecha", "ganaderia", "riego",
        "suelo", "fertilizante", "granja", "semilla", "campo"
    ],
    "alimentacion_nutricion": [
        "alimentacion", "nutricion", "dieta", "alimento", "proteina",
        "caloria", "vitamina", "receta", "ingrediente", "comida"
    ],
    "historia": [
        "historia", "siglo", "guerra", "revolucion", "imperio",
        "edad media", "civilizacion", "rey", "batalla", "colonizacion"
    ],
    "filosofia": [
        "filosofia", "etica", "moral", "existencia", "conocimiento",
        "razon", "metafisica", "epistemologia", "aristoteles", "platon"
    ],
    "literatura": [
        "literatura", "novela", "poesia", "cuento", "autor", "personaje",
        "narrador", "obra", "capitulo", "ensayo"
    ],
    "arte_cultura": [
        "arte", "cultura", "pintura", "escultura", "museo",
        "artista", "exposicion", "patrimonio", "estetica"
    ],
    "musica": [
        "musica", "cancion", "album", "ritmo", "melodia", "instrumento",
        "concierto", "compositor", "banda", "sonido"
    ],
    "cine_medios": [
        "cine", "pelicula", "serie", "guion", "director", "actor",
        "escena", "documental", "television", "medios"
    ],
    "deporte": [
        "deporte", "futbol", "baloncesto", "tenis", "entrenamiento",
        "partido", "liga", "jugador", "equipo", "competicion"
    ],
    "turismo_viajes": [
        "turismo", "viaje", "hotel", "destino", "vuelo", "reserva",
        "itinerario", "playa", "ciudad", "excursion"
    ],
    "geografia": [
        "geografia", "pais", "continente", "rio", "montaña", "mapa",
        "region", "clima", "territorio", "poblacion"
    ],
    "politica": [
        "politica", "gobierno", "partido", "elecciones", "parlamento",
        "presidente", "ministro", "democracia", "votacion"
    ],
    "administracion_publica": [
        "administracion", "publica", "ayuntamiento", "tramite",
        "licencia", "expediente", "subvencion", "boletin", "funcionario"
    ],
    "seguros": [
        "seguro", "poliza", "prima", "siniestro", "cobertura",
        "aseguradora", "indemnizacion", "riesgo", "beneficiario"
    ],
    "inmobiliario": [
        "inmobiliario", "vivienda", "alquiler", "compraventa",
        "hipoteca", "propiedad", "arrendamiento", "tasacion", "inmueble"
    ],
    "transporte": [
        "transporte", "tren", "avion", "barco", "autobus",
        "ruta", "pasajero", "mercancia", "billete", "logistica"
    ],
    "automocion": [
        "automocion", "coche", "vehiculo", "motor", "combustible",
        "electrico", "bateria", "concesionario", "reparacion"
    ],
    "telecomunicaciones": [
        "telecomunicaciones", "fibra", "movil", "5g", "internet",
        "antena", "operador", "red movil", "banda ancha"
    ],
    "videojuegos": [
        "videojuego", "juego", "gameplay", "consola", "jugador",
        "nivel", "personaje", "steam", "playstation", "xbox"
    ],
    "moda": [
        "moda", "ropa", "vestido", "marca", "tendencia", "calzado",
        "diseño", "textil", "coleccion", "estilo"
    ],
    "religion": [
        "religion", "iglesia", "dios", "fe", "biblia", "islam",
        "cristianismo", "budismo", "oracion", "creencia"
    ],
    "seguridad_laboral": [
        "prevencion", "riesgos laborales", "accidente laboral",
        "epi", "seguridad laboral", "salud laboral", "protocolo", "norma"
    ],
    "general": [
        "documento", "informacion", "contenido", "tema", "texto",
        "informe", "archivo", "resumen"
    ],
}


ALLOWED_TOPICS = list(TOPIC_KEYWORDS.keys())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text


def count_keyword_matches(text: str, keyword: str) -> int:
    keyword = normalize_text(keyword)

    if " " in keyword:
        return text.count(keyword)

    return len(re.findall(rf"\b{re.escape(keyword)}\b", text))


def classify_document(text: str, filename: str | None = None) -> str:
    filename = filename or ""
    filename_text = normalize_text(Path(filename).stem.replace("_", " "))

    normalized_text = normalize_text(text[:12000])
    full_text = f"{filename_text} {normalized_text}"

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            keyword_normalized = normalize_text(keyword)

            score += count_keyword_matches(full_text, keyword_normalized)

            if keyword_normalized in filename_text:
                score += 5

        scores[topic] = score

    best_topic = max(scores, key=scores.get)

    if scores[best_topic] == 0:
        return "general"

    return best_topic


def classify_question(question: str) -> str:
    return classify_document(question)