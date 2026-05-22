import re
import unicodedata
from pathlib import Path


UNKNOWN_TOPIC = "desconocida"

TOPIC_KEYWORDS = {
    "astronomia_universo": [
        "universo", "cosmos", "big bang", "galaxia", "galaxias", "via lactea",
        "estrella", "estrellas", "planeta", "planetas", "sistema solar",
        "agujero negro", "agujeros negros", "nebulosa", "supernova", "pulsar",
        "quasar", "astro", "astronomia", "cosmologia", "espacio", "espacial",
        "tiempo", "espaciotiempo", "telescopio", "observatorio", "satelite",
        "orbita", "gravedad", "gravitacion", "materia oscura", "energia oscura",
        "radiacion cosmica", "fondo cosmico", "asteroide", "cometa", "meteorito",
        "luna", "marte", "jupiter", "saturno", "exoplaneta", "constelacion",
        "hidrogeno", "helio", "quarks", "neutrinos", "electrones", "particulas"
    ],

    "ciencia_general": [
        "ciencia", "cientifico", "cientifica", "investigacion", "experimento",
        "experimental", "hipotesis", "teoria", "ley cientifica", "metodo cientifico",
        "observacion", "evidencia", "analisis", "resultado", "laboratorio",
        "ensayo", "prueba", "modelo teorico", "modelo experimental", "datos",
        "medicion", "muestra", "variable", "control", "conclusion", "publicacion",
        "revista cientifica", "articulo cientifico", "descubrimiento", "conocimiento",
        "fenomeno", "demostracion", "validacion", "reproducibilidad"
    ],

    "fisica": [
        "fisica", "mecanica", "cinematica", "dinamica", "fuerza", "masa",
        "energia", "trabajo", "potencia", "movimiento", "velocidad", "aceleracion",
        "gravedad", "gravitacion", "inercia", "newton", "relatividad",
        "cuantica", "mecanica cuantica", "onda", "ondas", "particula",
        "particulas", "campo", "campo electrico", "campo magnetico",
        "electricidad", "magnetismo", "optica", "luz", "fotones", "sonido",
        "termodinamica", "temperatura", "calor", "presion", "fluidos",
        "nuclear", "atomo", "electron", "proton", "neutron"
    ],

    "quimica": [
        "quimica", "quimico", "quimica organica", "quimica inorganica",
        "materia", "molecula", "molecular", "atomo", "atomico", "elemento",
        "elementos", "tabla periodica", "compuesto", "compuestos", "sustancia",
        "sustancias", "reaccion", "reacciones", "reaccion quimica",
        "enlace", "enlace ionico", "enlace covalente", "enlace metalico",
        "acido", "acidos", "base", "bases", "ph", "neutralizacion",
        "disolucion", "solucion", "concentracion", "mol", "molaridad",
        "estequiometria", "ion", "iones", "cation", "anion", "oxidacion",
        "reduccion", "redox", "electrolisis", "entalpia", "entropia",
        "equilibrio quimico", "catalizador", "precipitacion", "solubilidad",
        "alcohol", "alcoholes", "aldehido", "aldehidos", "ester", "esteres",
        "acido organico", "polimero", "polimeros", "hidrocarburo"
    ],

    "biologia": [
        "biologia", "biologico", "celula", "celulas", "adn", "arn", "gen",
        "genes", "genetica", "cromosoma", "proteina", "proteinas", "enzima",
        "enzimas", "organismo", "organismos", "tejido", "tejidos", "organo",
        "organos", "sistema nervioso", "sistema inmune", "evolucion",
        "seleccion natural", "ecosistema", "biodiversidad", "especie",
        "especies", "microorganismo", "bacteria", "bacterias", "virus",
        "hongos", "metabolismo", "fotosintesis", "respiracion celular",
        "mitosis", "meiosis", "mutacion", "herencia", "zoologia", "botanica"
    ],

    "medicina_salud": [
        "medicina", "medicina interna", "medico", "medica", "medicos",
        "paciente", "pacientes", "salud", "sanidad", "sanitario",
        "asistencia sanitaria", "diagnostico", "diagnosticar", "tratamiento",
        "tratamientos", "terapia", "terapeutico", "terapeutica", "hospital",
        "hospitalario", "clinica", "clinico", "enfermedad", "enfermedades",
        "enfermo", "enfermos", "curacion", "curar", "prevenir", "prevencion",
        "farmaco", "farmacos", "medicamento", "medicamentos", "cirugia",
        "cirujano", "internista", "internistas", "fisiologia", "anatomia",
        "patologia", "microbiologia", "infecciosas", "infeccion", "bioetica",
        "cuerpo humano", "sintoma", "sintomas", "dolor", "urgencias",
        "cuidados intensivos", "transplante", "enfermeria", "epidemiologia"
    ],

    "psicologia": [
        "psicologia", "psicologico", "mente", "conducta", "comportamiento",
        "emocion", "emociones", "ansiedad", "depresion", "estres", "trauma",
        "terapia", "psicoterapia", "cognitivo", "cognicion", "personalidad",
        "memoria", "aprendizaje", "motivacion", "percepcion", "autoestima",
        "trastorno", "trastornos", "salud mental", "psiquiatria",
        "psicoanalisis", "neuropsicologia", "desarrollo emocional",
        "inteligencia emocional", "conductual", "afectivo", "relacion social"
    ],

    "educacion": [
        "educacion", "educativo", "alumno", "alumna", "alumnos", "estudiante",
        "estudiantes", "profesor", "profesora", "docente", "maestro",
        "curso", "asignatura", "materia", "clase", "aula", "examen",
        "evaluacion", "calificacion", "aprendizaje", "ensenanza",
        "universidad", "colegio", "instituto", "bachillerato", "curriculo",
        "competencias", "objetivos", "contenidos", "didactica", "pedagogia",
        "formacion", "titulacion", "grado", "master", "educacion primaria",
        "educacion secundaria", "actividad educativa"
    ],

    "matematicas": [
        "matematicas", "matematico", "algebra", "calculo", "geometria",
        "aritmetica", "trigonometria", "integral", "derivada", "limite",
        "funcion", "funciones", "ecuacion", "ecuaciones", "inecuacion",
        "matriz", "matrices", "vector", "vectores", "probabilidad",
        "estadistica", "media", "varianza", "desviacion", "distribucion",
        "teorema", "demostracion", "numero", "numeros", "polinomio",
        "logaritmo", "exponencial", "serie", "sucesion", "topologia",
        "analisis matematico", "optimizacion"
    ],

    "tecnologia": [
        "tecnologia", "tecnologico", "digital", "software", "hardware",
        "sistema", "sistemas", "plataforma", "aplicacion", "aplicaciones",
        "dispositivo", "dispositivos", "servidor", "cloud", "nube",
        "internet", "datos", "base de datos", "automatizacion",
        "robotica", "sensor", "sensores", "electronica", "computacion",
        "informatica", "innovacion", "transformacion digital", "producto digital",
        "herramienta digital", "infraestructura tecnologica"
    ],

    "programacion": [
        "programacion", "programar", "codigo", "codificacion", "script",
        "python", "java", "javascript", "typescript", "html", "css",
        "sql", "api", "endpoint", "backend", "frontend", "fullstack",
        "funcion", "funciones", "clase", "objeto", "variable", "metodo",
        "framework", "libreria", "paquete", "bug", "debug", "error",
        "repositorio", "git", "github", "docker", "fastapi", "streamlit",
        "flask", "django", "node", "react", "vue", "angular"
    ],

    "inteligencia_artificial": [
        "inteligencia artificial", "ia", "ai", "machine learning",
        "aprendizaje automatico", "deep learning", "aprendizaje profundo",
        "modelo", "modelos", "red neuronal", "redes neuronales", "llm",
        "large language model", "rag", "embedding", "embeddings",
        "prompt", "fine tuning", "clasificador", "clasificacion",
        "regresion", "dataset", "entrenamiento", "inferencia", "token",
        "transformer", "openai", "chatgpt", "ollama", "langchain",
        "agente", "agentes", "nlp", "procesamiento de lenguaje natural",
        "vision artificial", "generativo", "modelo generativo"
    ],

    "ciberseguridad": [
        "ciberseguridad", "seguridad informatica", "vulnerabilidad",
        "vulnerabilidades", "malware", "ransomware", "phishing", "firewall",
        "cifrado", "encriptacion", "ataque", "ataques", "amenaza",
        "amenazas", "riesgo", "exploit", "brecha", "intrusion",
        "autenticacion", "autorizacion", "password", "contraseña",
        "hash", "token", "seguridad de red", "pentesting", "auditoria",
        "zero trust", "antivirus", "incidente", "forense", "privacidad",
        "proteccion de datos", "ddos", "vpn"
    ],

    "redes_sistemas": [
        "red", "redes", "sistemas", "ip", "direccion ip", "dns", "dhcp",
        "servidor", "cliente", "router", "switch", "firewall", "proxy",
        "linux", "windows server", "protocolo", "tcp", "udp", "http",
        "https", "ssh", "ftp", "puerto", "dominio", "hosting",
        "virtualizacion", "contenedor", "docker", "kubernetes",
        "administracion de sistemas", "monitorizacion", "logs", "backup",
        "infraestructura", "devops", "cluster", "balanceador"
    ],

    "legal": [
        "legal", "juridico", "juridica", "derecho", "ley", "leyes",
        "normativa", "reglamento", "contrato", "contratos", "clausula",
        "clausulas", "acuerdo", "firma", "obligacion", "obligaciones",
        "demanda", "sentencia", "tribunal", "juzgado", "abogado",
        "abogada", "licencia", "responsabilidad", "delito", "penal",
        "civil", "mercantil", "laboral", "administrativo", "arrendamiento",
        "propiedad intelectual", "proteccion de datos", "gdpr", "rgpd"
    ],

    "finanzas": [
        "finanzas", "financiero", "financiera", "factura", "facturacion",
        "balance", "ingresos", "gastos", "presupuesto", "inversion",
        "inversiones", "capital", "rentabilidad", "beneficio", "perdida",
        "activo", "pasivo", "liquidez", "flujo de caja", "cash flow",
        "banco", "prestamo", "credito", "interes", "deuda", "acciones",
        "bonos", "mercado financiero", "riesgo financiero", "dividendo",
        "patrimonio", "coste", "margen", "ebitda", "roi"
    ],

    "contabilidad": [
        "contabilidad", "contable", "asiento contable", "cuenta", "cuentas",
        "debe", "haber", "libro diario", "libro mayor", "balance contable",
        "activo", "pasivo", "patrimonio neto", "amortizacion",
        "depreciacion", "inventario", "iva", "impuesto", "fiscal",
        "auditoria", "cierre contable", "cuenta de resultados",
        "perdidas y ganancias", "factura", "proveedor", "cliente",
        "conciliacion bancaria", "registro contable"
    ],

    "economia": [
        "economia", "economico", "economica", "mercado", "oferta",
        "demanda", "precio", "inflacion", "deflacion", "pib",
        "producto interior bruto", "recesion", "crecimiento economico",
        "macroeconomia", "microeconomia", "empleo", "desempleo",
        "productividad", "competencia", "monopolio", "oligopolio",
        "politica monetaria", "politica fiscal", "tipo de interes",
        "banco central", "comercio", "exportacion", "importacion"
    ],

    "rrhh": [
        "recursos humanos", "rrhh", "empleado", "empleados", "trabajador",
        "trabajadores", "nomina", "vacaciones", "contratacion",
        "seleccion", "entrevista", "candidato", "puesto", "salario",
        "retribucion", "desempeno", "evaluacion del desempeno",
        "formacion", "talento", "clima laboral", "convenio", "baja",
        "alta", "despido", "jornada", "horario", "teletrabajo",
        "prevencion laboral", "onboarding"
    ],

    "marketing": [
        "marketing", "mercadotecnia", "campaña", "campanas", "marca",
        "branding", "publicidad", "seo", "sem", "redes sociales",
        "audiencia", "conversion", "conversiones", "posicionamiento",
        "cliente", "clientes", "segmentacion", "buyer persona",
        "email marketing", "contenido", "copywriting", "funnel",
        "embudo", "lead", "landing page", "analitica", "kpi",
        "engagement", "mercado objetivo"
    ],

    "ventas": [
        "ventas", "venta", "comercial", "cliente", "clientes", "lead",
        "prospecto", "crm", "pipeline", "embudo de ventas", "negociacion",
        "propuesta", "oferta", "presupuesto", "facturacion", "pedido",
        "contrato comercial", "cuota", "objetivo comercial", "comision",
        "captacion", "retencion", "upselling", "cross selling",
        "cierre", "oportunidad", "vendedor", "equipo comercial"
    ],

    "empresa_negocio": [
        "empresa", "negocio", "compania", "organizacion", "corporacion",
        "estrategia", "modelo de negocio", "plan de negocio", "kpi",
        "objetivo", "objetivos", "operaciones", "direccion", "gestion",
        "administracion", "liderazgo", "competitividad", "mercado",
        "cliente", "proveedor", "producto", "servicio", "procesos",
        "rentabilidad", "crecimiento", "innovacion", "emprendimiento",
        "startup", "socios", "accionistas"
    ],

    "proyectos": [
        "proyecto", "proyectos", "planificacion", "tarea", "tareas",
        "hito", "hitos", "cronograma", "gantt", "scrum", "kanban",
        "sprint", "backlog", "entregable", "alcance", "riesgo",
        "riesgos", "recurso", "recursos", "presupuesto", "stakeholder",
        "reunion", "seguimiento", "estado", "deadline", "fecha limite",
        "responsable", "roadmap", "gestion de proyectos"
    ],

    "logistica": [
        "logistica", "almacen", "almacenamiento", "inventario", "stock",
        "suministro", "cadena de suministro", "distribucion", "transporte",
        "pedido", "pedidos", "proveedor", "mercancia", "envio",
        "recepcion", "ruta", "flota", "operador logistico", "aduana",
        "importacion", "exportacion", "picking", "packing", "trazabilidad",
        "aprovisionamiento", "plazo de entrega"
    ],

    "industria_ingenieria": [
        "ingenieria", "industrial", "industria", "maquina", "maquinas",
        "proceso", "procesos", "fabricacion", "produccion", "planta",
        "mantenimiento", "automatizacion", "robot", "robotica",
        "calidad", "control de calidad", "linea de produccion",
        "manufactura", "materiales", "mecanica", "electrica",
        "electronica", "diseno tecnico", "prototipo", "ensamblaje",
        "eficiencia", "operacion industrial"
    ],

    "arquitectura_construccion": [
        "arquitectura", "construccion", "obra", "edificio", "vivienda",
        "plano", "planos", "estructura", "cemento", "hormigon",
        "ladrillo", "cimentacion", "fachada", "cubierta", "licencia",
        "urbanismo", "arquitecto", "ingeniero civil", "reforma",
        "presupuesto de obra", "materiales de construccion", "contratista",
        "seguridad en obra", "instalaciones", "proyecto arquitectonico"
    ],

    "energia_medioambiente": [
        "energia", "medioambiente", "medio ambiente", "sostenibilidad",
        "renovable", "renovables", "energia solar", "solar", "eolica",
        "hidroelectrica", "biomasa", "contaminacion", "emisiones",
        "co2", "carbono", "huella de carbono", "cambio climatico",
        "clima", "reciclaje", "residuos", "biodiversidad",
        "impacto ambiental", "eficiencia energetica", "descarbonizacion",
        "combustibles fosiles", "ecologia", "conservacion"
    ],

    "agricultura": [
        "agricultura", "agricola", "cultivo", "cultivos", "cosecha",
        "ganaderia", "riego", "suelo", "fertilizante", "semilla",
        "granja", "campo", "tractor", "plaga", "pesticida",
        "herbicida", "invernadero", "agronomia", "produccion agricola",
        "explotacion agraria", "cereal", "fruta", "verdura", "olivar",
        "vinicultura", "agricultura ecologica", "sostenibilidad agricola"
    ],

    "alimentacion_nutricion": [
        "alimentacion", "nutricion", "nutricional", "dieta", "dietas",
        "alimento", "alimentos", "comida", "proteina", "proteinas",
        "hidratos", "carbohidratos", "grasas", "vitamina", "vitaminas",
        "mineral", "caloria", "calorias", "receta", "ingrediente",
        "ingredientes", "cocina", "gastronomia", "salud alimentaria",
        "obesidad", "peso", "metabolismo", "suplemento", "menu"
    ],

    "historia": [
        "historia", "historico", "siglo", "edad media", "edad antigua",
        "edad moderna", "edad contemporanea", "guerra", "batalla",
        "revolucion", "imperio", "civilizacion", "rey", "reina",
        "monarquia", "republica", "colonizacion", "independencia",
        "arqueologia", "prehistoria", "roma", "grecia", "egipto",
        "feudalismo", "renacimiento", "ilustracion", "cronologia"
    ],

    "filosofia": [
        "filosofia", "filosofo", "etica", "moral", "metafisica",
        "epistemologia", "conocimiento", "razon", "existencia", "ser",
        "verdad", "alma", "conciencia", "libertad", "justicia",
        "pensamiento", "argumento", "logica", "aristoteles", "platon",
        "kant", "descartes", "nietzsche", "socrates", "estoicismo",
        "utilitarismo", "racionalismo", "empirismo"
    ],

    "literatura": [
        "literatura", "literario", "novela", "poesia", "poema", "cuento",
        "relato", "ensayo", "autor", "autora", "personaje", "narrador",
        "trama", "capitulo", "obra", "genero literario", "teatro",
        "drama", "comedia", "verso", "prosa", "metafora", "simbolo",
        "romanticismo", "realismo", "modernismo", "critica literaria"
    ],

    "arte_cultura": [
        "arte", "cultura", "cultural", "pintura", "escultura", "museo",
        "artista", "exposicion", "patrimonio", "estetica", "galeria",
        "obra de arte", "dibujo", "fotografia", "arquitectura artistica",
        "historia del arte", "vanguardia", "renacimiento", "barroco",
        "impresionismo", "arte moderno", "arte contemporaneo", "tradicion",
        "identidad cultural", "manifestacion cultural"
    ],

    "musica": [
        "musica", "musical", "cancion", "canciones", "album", "ritmo",
        "melodia", "armonia", "instrumento", "instrumentos", "concierto",
        "compositor", "banda", "grupo musical", "sonido", "voz",
        "guitarra", "piano", "violin", "bateria", "orquesta",
        "sinfonia", "genero musical", "jazz", "rock", "pop",
        "clasica", "flamenco", "produccion musical"
    ],

    "cine_medios": [
        "cine", "pelicula", "peliculas", "serie", "series", "guion",
        "director", "actriz", "actor", "escena", "documental",
        "television", "medios", "audiovisual", "rodaje", "camara",
        "montaje", "produccion", "productora", "critica cinematografica",
        "streaming", "plataforma", "episodio", "temporada", "trailer",
        "ficcion", "entretenimiento", "comunicacion"
    ],

    "deporte": [
        "deporte", "deportivo", "futbol", "baloncesto", "tenis",
        "atletismo", "natacion", "ciclismo", "entrenamiento", "partido",
        "competicion", "liga", "torneo", "jugador", "jugadora", "equipo",
        "entrenador", "gol", "puntos", "clasificacion", "rendimiento",
        "preparacion fisica", "lesion deportiva", "olimpiadas",
        "campeonato", "arbitro", "estadio"
    ],

    "turismo_viajes": [
        "turismo", "viaje", "viajes", "hotel", "hoteles", "destino",
        "destinos", "vuelo", "vuelos", "reserva", "itinerario",
        "excursion", "playa", "ciudad", "ruta turistica", "guia",
        "agencia de viajes", "alojamiento", "maleta", "pasaporte",
        "visado", "aeropuerto", "tren", "crucero", "vacaciones",
        "turista", "patrimonio", "gastronomia local"
    ],

    "geografia": [
        "geografia", "geografico", "pais", "paises", "continente",
        "continentes", "rio", "rios", "montaña", "montanas", "mapa",
        "region", "territorio", "poblacion", "clima", "relieve",
        "latitud", "longitud", "frontera", "capital", "ciudad",
        "cordillera", "valle", "desierto", "oceano", "mar",
        "hidrografia", "demografia", "cartografia"
    ],

    "politica": [
        "politica", "politico", "gobierno", "partido", "partidos",
        "elecciones", "votacion", "voto", "parlamento", "congreso",
        "senado", "presidente", "ministro", "democracia", "estado",
        "constitucion", "ideologia", "campaña electoral", "coalicion",
        "oposicion", "ley", "reforma", "administracion", "poder ejecutivo",
        "poder legislativo", "politica publica"
    ],

    "administracion_publica": [
        "administracion publica", "administracion", "publica",
        "ayuntamiento", "ministerio", "tramite", "tramites", "licencia",
        "expediente", "subvencion", "boletin", "funcionario",
        "funcionarios", "servicio publico", "contratacion publica",
        "licitacion", "concurso publico", "resolucion", "normativa",
        "ciudadano", "sede electronica", "procedimiento administrativo"
    ],

    "seguros": [
        "seguro", "seguros", "poliza", "prima", "siniestro", "cobertura",
        "aseguradora", "indemnizacion", "riesgo", "beneficiario",
        "tomador", "asegurado", "responsabilidad civil", "seguro de vida",
        "seguro de salud", "seguro de hogar", "seguro de coche",
        "franquicia", "peritaje", "reclamacion", "contrato de seguro"
    ],

    "inmobiliario": [
        "inmobiliario", "inmueble", "vivienda", "piso", "casa",
        "alquiler", "arrendamiento", "compraventa", "hipoteca",
        "propiedad", "tasacion", "catastro", "registro de la propiedad",
        "arrendador", "arrendatario", "fianza", "contrato de alquiler",
        "agencia inmobiliaria", "terreno", "local", "edificio",
        "comunidad de propietarios", "renta", "precio de venta"
    ],

    "transporte": [
        "transporte", "tren", "avion", "barco", "autobus", "metro",
        "taxi", "ruta", "pasajero", "pasajeros", "mercancia",
        "billete", "logistica", "flota", "vehiculo", "carretera",
        "aeropuerto", "puerto", "estacion", "trafico", "movilidad",
        "transporte publico", "transporte privado", "carga", "envio"
    ],

    "automocion": [
        "automocion", "coche", "coches", "vehiculo", "vehiculos",
        "motor", "combustible", "gasolina", "diesel", "electrico",
        "hibrido", "bateria", "concesionario", "reparacion", "taller",
        "neumatico", "freno", "embrague", "transmision", "aceite",
        "matricula", "itv", "seguro de coche", "kilometraje",
        "fabricante", "modelo", "chasis"
    ],

    "telecomunicaciones": [
        "telecomunicaciones", "fibra", "fibra optica", "movil",
        "telefonia", "5g", "4g", "internet", "antena", "operador",
        "red movil", "banda ancha", "router", "wifi", "tarifa",
        "datos moviles", "cobertura", "senal", "latencia",
        "comunicacion", "llamada", "sms", "satelite", "radiofrecuencia"
    ],

    "videojuegos": [
        "videojuego", "videojuegos", "juego", "juegos", "gameplay",
        "consola", "jugador", "jugadores", "nivel", "personaje",
        "steam", "playstation", "xbox", "nintendo", "pc gaming",
        "multijugador", "online", "partida", "desarrollador",
        "motor grafico", "unity", "unreal", "e-sports", "skin",
        "mision", "enemigo", "mapa"
    ],

    "moda": [
        "moda", "ropa", "vestido", "camisa", "pantalon", "calzado",
        "zapato", "marca", "tendencia", "estilo", "diseño",
        "textil", "coleccion", "pasarela", "modelo", "tejido",
        "algodon", "lana", "cuero", "accesorio", "bolso",
        "joyeria", "estilismo", "temporada", "modista"
    ],

    "religion": [
        "religion", "religioso", "iglesia", "dios", "fe", "biblia",
        "evangelio", "cristianismo", "islam", "budismo", "judaismo",
        "oracion", "creencia", "espiritualidad", "sagrado", "sacerdote",
        "mezquita", "templo", "monasterio", "ritual", "culto",
        "teologia", "pecado", "salvacion", "profeta"
    ],

    "seguridad_laboral": [
        "seguridad laboral", "prevencion", "riesgos laborales",
        "accidente laboral", "epi", "equipo de proteccion individual",
        "salud laboral", "protocolo", "norma", "normativa laboral",
        "evaluacion de riesgos", "plan de prevencion", "siniestralidad",
        "ergonomia", "higiene industrial", "proteccion", "formacion preventiva",
        "riesgo", "peligro", "trabajador", "inspeccion", "mutua"
    ],
}
UNKNOWN_TOPIC = "desconocida"
ALLOWED_TOPICS = list(TOPIC_KEYWORDS.keys()) + [UNKNOWN_TOPIC]

DOCUMENT_MIN_SCORE = 3
QUESTION_MIN_SCORE = 1
MIN_SCORE_MARGIN = 1


def normalize_text(text: str) -> str:
    text = text or ""
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


def get_topic_scores(text: str, filename: str | None = None) -> dict[str, int]:
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

    return scores


def classify_with_threshold(
    text: str,
    filename: str | None = None,
    min_score: int = DOCUMENT_MIN_SCORE,
) -> str:
    scores = get_topic_scores(text, filename)

    if not scores:
        return UNKNOWN_TOPIC

    ranked_topics = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_topic, best_score = ranked_topics[0]
    second_score = ranked_topics[1][1] if len(ranked_topics) > 1 else 0

    if best_score < min_score:
        return UNKNOWN_TOPIC

    if second_score > 0 and (best_score - second_score) < MIN_SCORE_MARGIN:
        return UNKNOWN_TOPIC

    return best_topic


def classify_document(text: str, filename: str | None = None) -> str:
    return classify_with_threshold(
        text=text,
        filename=filename,
        min_score=DOCUMENT_MIN_SCORE,
    )


def classify_question(question: str) -> str:
    return classify_with_threshold(
        text=question,
        filename=None,
        min_score=QUESTION_MIN_SCORE,
    )