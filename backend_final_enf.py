from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import re
from datetime import datetime, timedelta
import dateparser
from dateparser.search import search_dates
import csv
from wordcloud import WordCloud
from flask import send_file
from collections import OrderedDict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from email.utils import formataddr
from dotenv import load_dotenv
from babel.dates import format_date
from s3_utils import s3_download_all as r2_download_all, s3_upload as r2_upload
import faiss
import requests
# LangChain / RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import hashlib
import json

print("✅ CARGÓ backend_final_enf.py — VERSION 2025-12-30 17:xx")

def quitar_markdown_basico(texto: str) -> str:
    if not texto:
        return texto

    # Quitar negritas Markdown
    texto = texto.replace("**", "")

    # Convertir bullets tipo "* " a guiones normales
    texto = re.sub(r"(?m)^\s*\*\s+", "- ", texto)

    return texto

MAX_TITULARES_SELECCION = 10

def nombre_mes(fecha):
    """Devuelve la fecha con mes en español, ej: 'agosto 2025'"""
    return format_date(fecha, "LLLL yyyy", locale="es").capitalize()


# ------------------------------
# 🔑 Configuración API y Flask
# ------------------------------
load_dotenv()
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 🔄 Sincronizar índices y metadatos desde Cloudflare R2 al iniciar
try:
    r2_download_all()
    print("✅ Archivos FAISS/CSV sincronizados desde R2")
except Exception as e:
    print(f"⚠️ No se pudo sincronizar desde R2: {e}")

@app.route("/")
def home():
    return send_file("index.html")
# ------------------------------
# 📂 Carga única de datos — con rutas absolutas seguras
base_dir = os.path.dirname(os.path.abspath(__file__))

print("📁 Base directory:", base_dir)

# --- Cargar base de noticias ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noticias_path = os.path.join(base_dir, "noticias_enfragen.csv")
    print("📁 Base directory:", base_dir)
    print("Intentando leer:", noticias_path)

    df = pd.read_csv(noticias_path, encoding="utf-8")
    print(f"✅ Noticias cargadas: {len(df)} filas")
    print("🧩 Columnas detectadas:", list(df.columns))

    # Detectar automáticamente la columna de fecha
    fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
    if fecha_col:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=True)
        df = df.rename(columns={fecha_col: "Fecha"}).dropna(subset=["Fecha"])
        print(f"📅 Columna '{fecha_col}' convertida correctamente. Rango:",
              df["Fecha"].min(), "→", df["Fecha"].max())
    else:
        print("⚠️ No se encontró columna con 'fecha' en el nombre.")
        df["Fecha"] = pd.NaT
# 🔗 ---------- LangChain: embeddings, vectorstores y LLM ----------

    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
)

    vectorstore_noticias = None
    retriever_noticias = None

    vectorstore_resumenes = None
    retriever_resumenes = None

    # ------------------------------
    # 🔗 MODELO LLM Y CHAIN PARA /pregunta
    # ------------------------------

    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
    )

    prompt_pregunta = ChatPromptTemplate.from_messages([
        ("system", """
Eres un analista experto en ENERGÍA, con éfasis en el contexto colombiano.
Responde SIEMPRE en español.
NO inventes datos ni traigas información de fuera del contexto.
Si el contexto incluye al menos un titular o un resumen relevante, NO digas que “no se dispone de información” ni frases parecidas; en su lugar, explica lo que SÍ se sabe con base en esos elementos.
Solo si el contexto está totalmente vacío (sin titulares ni resúmenes sobre el tema) puedes decir que no hay información disponible.
Tu objetivo es responder la pregunta del usuario de forma profesional, clara y basada en los titulares y resúmenes proporcionados.
REGLAS FUNDAMENTALES (PROHIBICIONES ABSOLUTAS)
- Está TERMINANTEMENTE PROHIBIDO:
  - Explicar por qué algo es importante, relevante, significativo o preocupante.
  - Usar frases como:
    “lo que implica”, “lo que refuerza”, “lo que podría”, “lo que resalta”, “esto es clave”, “esto podría ser”.
  - Hacer inferencias, conclusiones, evaluaciones o lecturas políticas.
Reglas adicionales: 
  - NO expliques consecuencias.
  - NO relaciones hechos entre sí si los titulares no lo hacen explícitamente.
"""),
        ("user", "{texto_usuario}")
    ])


    chain_pregunta = prompt_pregunta | llm_chat | StrOutputParser()
    

    def cargar_vectorstore_noticias(df_noticias: pd.DataFrame):
        """
        Construye o actualiza de forma incremental el vectorstore de noticias.

        - Primera vez: embebe todas las noticias y crea el índice.
        - Siguientes veces: detecta qué filas del df no están todavía embebidas
        (por clave única) y solo calcula embeddings para esas noticias nuevas.
        IMPORTANTE:
        Ya no se usa información de cobertura geográfica ni de idioma. El foco
        está en título, fecha, fuente, enlace, término y sentimiento.
        
        """
        global vectorstore_noticias, retriever_noticias

        if df_noticias is None or df_noticias.empty:
            print("⚠️ df_noticias vacío, no se construye vectorstore_noticias")
            vectorstore_noticias = None
            retriever_noticias = None
            return

        # 📁 Directorio base para guardar índice y metadatos de LangChain
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(base_dir, "faiss_index", "noticias_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_path = os.path.join(index_dir, "noticias_lc_metadata.csv")

        # 1️⃣ Construir clave única para cada noticia del df actual
        df_noticias = df_noticias.copy()

        def make_unique_key(row):
            titulo = str(row.get("Título", "")).strip()
            fuente = str(row.get("Fuente", "")).strip()
            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_iso = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_iso = ""
            else:
                fecha_iso = ""
            return f"{fecha_iso}|{fuente}|{titulo}"

        df_noticias["unique_key_lc"] = df_noticias.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos (si existen) para saber qué noticias ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_path):
            try:
                df_meta_prev = pd.read_csv(meta_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos cargados: {len(existing_keys)} noticias embebidas.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de noticias: {e}")
                df_meta_prev = None
                existing_keys = set()

        # 3️⃣ Detectar noticias nuevas (filas cuyo unique_key_lc no está en existing_keys)
        mask_new = ~df_noticias["unique_key_lc"].isin(existing_keys)
        df_new = df_noticias[mask_new].copy()

        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Término", "Sentimiento"
            ])

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_noticias = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_noticias = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_noticias existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_noticias existente, se reconstruirá desde cero: {e}")
                vectorstore_noticias = None

        # 5️⃣ Construir Document para noticias nuevas
        docs_nuevos = []
        for _, row in df_new.iterrows():
            titulo = str(row.get("Título", "")).strip()
            if not titulo:
                continue

            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_str = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_str = None
            else:
                fecha_str = None

            metadata = {
                "fecha": fecha_str,
                "fuente": row.get("Fuente"),
                "enlace": row.get("Enlace"),
                "sentimiento": row.get("Sentimiento"),
                "termino": row.get("Término"),
                "unique_key_lc": row.get("unique_key_lc"),
            }

            docs_nuevos.append(Document(page_content=titulo, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de noticias
        if vectorstore_noticias is None:
            # Primera vez: si no hay índice previo, construirlo desde cero con TODO lo nuevo
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_noticias desde cero con {len(docs_nuevos)} noticias…")
                vectorstore_noticias = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay documentos nuevos y no existe índice previo; no se construye vectorstore_noticias.")
                retriever_noticias = None
                return
        else:
            # Ya había índice previo: solo agregamos los documentos nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} noticias nuevas a vectorstore_noticias…")
                vectorstore_noticias.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay noticias nuevas para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Término", "Sentimiento"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        try:
            df_meta_final.to_csv(meta_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de noticias guardados/actualizados en {meta_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de noticias: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_noticias.save_local(index_dir)
            print(f"✅ vectorstore_noticias guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_noticias: {e}")

        # 8.1️⃣ Subir índice y metadatos de noticias a S3
        try:
            # Subir CSV de metadatos de noticias
            rel_meta_key = os.path.join("noticias_lc", "noticias_lc_metadata.csv")
            r2_upload(rel_meta_key)

            # Subir archivos principales del índice FAISS de LangChain
            for fname in ["index.faiss", "index.pkl"]:
                rel_key = os.path.join("noticias_lc", fname)
                r2_upload(rel_key)

            print("☁️ Índice de noticias y metadatos subidos a S3.")
        except Exception as e:
            print(f"⚠️ No se pudo subir índice de noticias a S3: {e}")

        # 9️⃣ Crear el retriever
        retriever_noticias = vectorstore_noticias.as_retriever(search_kwargs={"k": 8})
        print("✅ retriever_noticias listo para usarse.")



    def cargar_vectorstore_resumenes():
        """
        Construye o actualiza de forma incremental el vectorstore de resúmenes.

        - Primera vez: embebe todos los resúmenes presentes en faiss_index/resumenes_metadata.csv
        y crea un índice específico para LangChain.
        - Siguientes veces: detecta qué resúmenes son nuevos (por clave única) y solo calcula embeddings
        para esos resúmenes adicionales, agregándolos al índice existente.
        """
        global vectorstore_resumenes, retriever_resumenes

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 📁 CSV de origen con la info de los resúmenes (tu pipeline actual)
        origen_path = os.path.join(base_dir, "faiss_index", "resumenes_metadata.csv")
        if not os.path.exists(origen_path):
            print(f"⚠️ No se encontró {origen_path}, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        try:
            df_origen = pd.read_csv(origen_path, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error al leer {origen_path}: {e}")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        if df_origen.empty:
            print("⚠️ resumenes_metadata.csv está vacío, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        # Asegurar columnas esperadas mínimas
        for col in ["fecha", "resumen"]:
            if col not in df_origen.columns:
                print(f"⚠️ La columna '{col}' no está en resumenes_metadata.csv")
                vectorstore_resumenes = None
                retriever_resumenes = None
                return

        # 📁 Directorio para el índice y metadatos específicos de LangChain
        index_dir = os.path.join(base_dir, "faiss_index", "resumenes_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_lc_path = os.path.join(index_dir, "resumenes_lc_metadata.csv")

        # 1️⃣ Crear clave única para cada resumen (por ejemplo: fecha|archivo_txt)
        df_origen = df_origen.copy()

        def make_unique_key(row):
            fecha_val = str(row.get("fecha", "")).strip()
            archivo_txt = str(row.get("archivo_txt", "")).strip()
            if not archivo_txt:
                # Si no hay nombre de archivo, usamos solo fecha como clave
                return fecha_val
            return f"{fecha_val}|{archivo_txt}"

        df_origen["unique_key_lc"] = df_origen.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos de LangChain (si existen) para saber qué resúmenes ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_lc_path):
            try:
                df_meta_prev = pd.read_csv(meta_lc_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos de resúmenes cargados: {len(existing_keys)} embebidos.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de resúmenes: {e}")
                df_meta_prev = None
                existing_keys = set()
        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ])

        # 3️⃣ Detectar resúmenes nuevos (clave única no vista antes)
        mask_new = ~df_origen["unique_key_lc"].isin(existing_keys)
        df_new = df_origen[mask_new].copy()

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_resumenes = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_resumenes = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_resumenes existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_resumenes existente, se reconstruirá desde cero: {e}")
                vectorstore_resumenes = None

        # 5️⃣ Crear Document para resúmenes nuevos
        docs_nuevos = []
        for _, row in df_new.iterrows():
            texto = str(row.get("resumen", "")).strip()
            if not texto:
                continue

            fecha_meta = str(row.get("fecha", "")).strip() or None
            archivo_txt = str(row.get("archivo_txt", "")).strip() or None
            nube = str(row.get("nube", "")).strip() or None
            titulares = row.get("titulares", None)
            unique_key = row.get("unique_key_lc")

            metadata = {
                "fecha": fecha_meta,
                "archivo_txt": archivo_txt,
                "nube": nube,
                "titulares": titulares,
                "tipo": "resumen",
                "unique_key_lc": unique_key,
            }

            docs_nuevos.append(Document(page_content=texto, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de resúmenes
        if vectorstore_resumenes is None:
            # Primera vez: construimos el índice solo con los docs nuevos (que en la práctica serán todos)
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_resumenes desde cero con {len(docs_nuevos)} resúmenes…")
                vectorstore_resumenes = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay resúmenes nuevos y no existe índice previo; no se construye vectorstore_resumenes.")
                retriever_resumenes = None
                return
        else:
            # Ya había índice previo: solo agregamos los resúmenes nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} resúmenes nuevos a vectorstore_resumenes…")
                vectorstore_resumenes.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay resúmenes nuevos para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos de LangChain y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        # 7️⃣ Guardar metadatos de resúmenes
        try:
            df_meta_final.to_csv(meta_lc_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de resúmenes guardados/actualizados en {meta_lc_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de resúmenes: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_resumenes.save_local(index_dir)
            print(f"✅ vectorstore_resumenes guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_resumenes: {e}")

        # 8.1️⃣ Subir índice y metadatos de resúmenes a S3
        try:
            # CSV de metadatos de resúmenes (LangChain)
            rel_meta_key = os.path.join("resumenes_lc", "resumenes_lc_metadata.csv")
            r2_upload(rel_meta_key)

            # Archivos principales del índice FAISS de LangChain
            for fname in ["index.faiss", "index.pkl"]:
                rel_key = os.path.join("resumenes_lc", fname)
                r2_upload(rel_key)

            print("☁️ Índice de resúmenes y metadatos subidos a S3.")
        except Exception as e:
            print(f"⚠️ No se pudo subir índice de resúmenes a S3: {e}")

        # 9️⃣ Crear el retriever
        retriever_resumenes = vectorstore_resumenes.as_retriever(search_kwargs={"k": 3})
        print("✅ retriever_resumenes listo para usarse.")


    # ==========================================
    # 🧩 Inicializar Vectorstores (RAG)
    # ==========================================
    print("⚙️ Inicializando vectorstore de noticias...")
    cargar_vectorstore_noticias(df)

    print("⚙️ Inicializando vectorstore de resúmenes...")
    cargar_vectorstore_resumenes()

except Exception as e:
    print(f"❌ Error al cargar CSV de noticias: {e}")
    df = pd.DataFrame()


# 🧹 Utilidad para sanear JSON (convierte NaN/inf a None y numpy → tipos nativos)
def _json_sanitize(x):
    import math, numpy as np
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ _json_sanitize(v) for v in x ]
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

# ------------------------------
# 🔧 Modo de selección de titulares
# ------------------------------

# ------------------------------
# 📜 Contexto político único
# ------------------------------
CONTEXTO_POLITICO = """
GLOSARIO POLÍTICO
- Donald Trump es el actual Presidente de EE.UU, fue reelecto en 2024.
- Francia Márquez es la actual Vicepresidenta de Colombia, electa en 2022.
- Armando Benedetti es el actual Ministro de Interior de Colombia, desde el 1 de marzo de 2025.
- Edwin Palma es el actual Ministro de Minas y Energía de Colombia, en el cargo desde el 25 de febrero de 2025. 
- Gustavo Petro es el actual Presidente de Colombia, en el cargo desde 2022 y hasta 2026.
- Nicolás Petro es el hijo mayor de Gustavo Petro, quien está en medio de un proceso judicial bajo las acusaciones de lavado de dinero y enriquecimiento ilícito proveniente del narcotráfico el cual tenía como objetivo el financiamiento de la campaña presidencial de su padre.
CONTEXTO DEL SECTOR ENERGÉTICO COLOMBIANO
Este contexto sirve únicamente para “quién es quién” y definiciones. NO debe usarse para inventar hechos nuevos.
Si los titulares del día contradicen o actualizan algo, manda lo que dicen los titulares del día.
5 CLAVES DEL SECTOR ENERGÉTICO COLOOMBIANO
1) Dependencia de la hidroelectricidad: 70% de la electricidad proviene de hidroeléctricas. Fenómenos como El Niño y sequías elevan el riesgo; las térmicas (como las operadas por compañías de respaldo como EnfraGen) son críticas para la estabilidad cuando bajan los aportes hídricos.
2) Riesgo de desabastecimiento de gas natural: reservas probadas caen; retrasos en exploración y desarrollo; mayor dependencia de importaciones; regasificación avanza de forma heterogénea. Desde 2026 puede haber escasez para demanda esencial (hogares, industria y generación).
3) Ecopetrol como eje estructural: principal empresa de petróleo y gas, controlada mayoritariamente por el Estado (88.49%). Su estrategia y finanzas impactan seguridad energética y finanzas públicas.
4) Sistema tarifario con subsidios y diferimientos: subsidios a usuarios vulnerables + “Opción Tarifaria” (diferimiento de alzas). Hacia finales de 2025 se reporta un rezago > 6.1 billones COP con empresas del sector, afectando liquidez e inversión.
5) Alta exposición social/territorial: Afinia, Celsia, Cens, CEO (Compañía Energética de Occidente), Enel Colombia, EPM y Essa operan en territorios con pérdidas no técnicas, rezago en redes, tensiones tarifarias y vulnerabilidad social (especialmente Caribe). Sector frágil e interdependiente; gobierno con deudas y elecciones presidenciales en 2026.

QUIÉN ES QUIÉN / ACTORES CLAVE
- Ecopetrol: empresa ancla del sector (petróleo y gas), clave fiscal y energética.
- Ricardo Roa: presidente de Ecopetrol (desde abril 2023), exgerente de campaña Petro 2022. Cobertura reciente incluye presiones reputacionales/judiciales y debates sobre activos estratégicos y disputas tributarias.
- Ministerio de Minas y Energía (MinMinas): cabeza del sector desde el gobierno central (política pública, transición, tarifas, subsidios, gas, hidrocarburos y electricidad).
- Edwin Palma Egea: ministro (según el documento), abogado laboral y exdirigente sindical (USO). Cobertura reciente gira en torno a perfil político-sindical, gestión/defensa ante alertas y controversias/investigaciones (incluida gestión en Air-e).
- Comisión de Regulación de Energía y Gas o CREG: regulador técnico de reglas tarifarias y de mercado (electricidad y gas). En cobertura reciente aparece por medidas ante escasez de gas, revisión de estructura tarifaria y reglas especiales para empresas intervenidas.
- Air-e: distribuidora intervenida por el gobierno, símbolo de crisis histórica en Caribe (redes, pérdidas, morosidad, conflictividad social, dependencia de subsidios). En el documento: tras intervención se suspendieron pagos (incluyendo a EnfraGen), con debate regulatorio/jurídico.

GAS / REGASIFICACIÓN
- Crisis de gas: reservas probadas han caído fuertemente; mayor participación de gas importado; escenarios con alzas tarifarias relevantes.
- Sirius (offshore Caribe): proyecto estratégico (Petrobras + Ecopetrol) con retos (consultas previas, licencias, ritmos).
- Regasificación y Coveñas: puente de corto plazo; Coveñas (Cenit/Ecopetrol) con autorización ambiental e inversión anunciada; cuello de botella: unidad FSU.

DEUDA DEL GOBIERNO CON EL SECTOR (FIN 2025)
Rezago reportado: 6.1 billones COP:
- 2.4 billones Opción Tarifaria
- 2.3 billones subsidios
- 1.4 billones obligaciones de usuarios oficiales

NUEVOS DESARROLLOS:
- El 25 de diciembre de 2025, la Agencia Nacional de Hidrocarburos formalizó a favor de Ecopetrol la cesión del 50 % de la participación que tenía Shell EP Offshore Venture en los contratos de exploración y producción COL 5, Purple Angel y Fuerte Sur, en el caribe sur. Con esta decisión, Ecopetrol asume control total de los proyectos de exploración y producción de gas en el Caribe colombiano.  
- Ecopetrol ofrecerá en venta entre 39 y 68 millones de barriles diarios de gas natural, de enero a mayo de 2026. La Empresa Colombiana de Petróleos (Ecopetrol) anunció que pondrá a disposición del mercado entre 39 y 68 Gigas BTU por día (GBTUD)) de gas natural para los meses entre febrero y mayo de 2026, como parte de su estrategia para contribuir al abastecimiento del energético en el país. La compañía informó que el gas provendrá de los campos Cupiagua, Cupiagua Sur y Pauto Sur, ubicados en el Piedemonte Llanero.
- El 30 de diciembre de 2025, Ecopetrol inició el proceso de comercialización del gas natural que será importado a través del Terminal Marítimo de Coveñas. Se trata de una cantidad de 110 GBTUD (gigabritish thermal units por día) y con vigencia de 10 años. Este gas será entregado a través del sistema Coveñas – Ayacucho, que se conectará al gasoducto Ballena – Barrancabermeja, en el Sistema Nacional de Transporte (SNT), a partir del primer trimestre de 2027. Los contratos se ofrecerán en la modalidad “firme sujeto a condiciones”, conforme a la regulación vigente, en lo que se incorpora flexibilidad para modificar la fuente y el punto de entrega de gas según avancen los proyectos offshore en el Caribe sur.
- En el municipio de Yondó, Antioquia, comenzó a operar una granja solar impulsada por Ecopetrol. El proyecto, conocido como La Iguana, se levanta en un área de 26 hectáreas y está compuesto por más de 43.000 paneles solares. La infraestructura alcanza una capacidad instalada cercana a los 26 megavatios, suficiente para abastecer anualmente el consumo eléctrico de aproximadamente 23.500 hogares. La energía generada, además, será destinada a las operaciones de la refinería de Barrancabermeja y a los campos de producción de Llanito y Cazabe, ubicados en el Magdalena Medio.
- Desde este 1 de enero de 2026 entró en vigor un nuevo ajuste en los precios de la gasolina corriente y el ACPM (diésel) en Colombia, decretado por la Comisión de Regulación de Energía y Gas (Creg). El incremento promedio fue de aproximadamente $90 por galón de gasolina y $99 por galón de ACPM. En las 13 ciudades principales, el precio promedio de la gasolina está alrededor de $16.057 por galón, mientras que el ACPM promedia $10.984 por galón, según el listado oficial divulgado por la Creg.
- El 31 de diciembre se anunció que el Gobierno nacional prepara una intervención en el mercado del gas natural del país, en medio del incremento de precios que se ha presentado en los últimos años, según un proyecto de decreto del Ministerio de Minas que fue sometido a comentarios del público. La intervención consistiríra: en primer lugar, mientras el potencial de producción en el país sea apenas 1,15 veces superior a la demanda, el mercado tendrá condiciones especiales y quienes revendan gas deberán hacerlo prácticamente al mismo precio al que lo compran (solo se les permitirá ganar la DTF a 90 días). En segundo lugar, los productores de gas natural no podrán ofrecer gas interrumpible a precios superiores a los contratos que ya tienen vigentes en el mercado.
- El 31 de diciembre de 2025 se presentó una explosión en el municipio de Puente Nacional, Santander, tras un incidente en la infraestructura de Cenit que provocó un incendio de diésel y la propagación del fuego hacia una zona forestal. De acuerdo con Ecopetrol, la explosión fue ocasionada por recipientes ilegales usados para el almacenamiento de hidrocarburo hurtado. El hecho fue controlado sin afectar el poliducto.
- EPM dio un nuevo paso en el proceso de enajenación de las acciones que mantiene en UNE EPM Telecomunicaciones S. A. Con esta decisión, la empresa busca avanzar en la venta de las acciones remanentes que no fueron adjudicadas durante la primera fase. Según informó la empresa, en esta segunda etapa podrán participar personas naturales o jurídicas, tanto nacionales como extranjeras, que cumplan con los requisitos establecidos en el Reglamento de la Segunda Etapa, especialmente el proceso de precalificación que inició el 15 de noviembre de 2025.Durante la primera etapa, que estuvo vigente por dos meses y estuvo dirigida a los destinatarios de condiciones especiales, como trabajadores activos y pensionados, asociaciones de empleados o exempleados, sindicatos, fondos de empleados, fondos de cesantías y pensiones y entidades cooperativas, EPM adjudicó un total de 77 acciones.
- De acuerdo con una publicación de la emisora colombiana Blu Radio, los trabajadores de la Electrificadora de Santander (ESSA), filial del Grupo EPM, iniciaron una asamblea permanente luego de varias semanas de conversaciones sin acuerdos con la administración de la empresa.La medida fue adoptada por los sindicatos como respuesta a lo que califican como ausencia de avances en la negociación colectiva y a la persistencia de esquemas de tercerización laboral que afectan a una parte significativa del personal.De acuerdo con la información entregada por los representantes de los trabajadores, cerca de 3.500 empleados se encuentran vinculados de manera indirecta a la electrificadora, pese a desempeñar labores consideradas misionales.
- Con dos nuevas comisionadas, la CREG vuelve a tener quorum para sesionar.Después de más de medio año sin su conformación completa, los nombramientos de Ángela Patricia Álvarez Gutiérrez y Adriana María Jiménez Delgado permiten que la entidad vuelva a estar habilitada para tomar decisiones en energía y gas.
- La Autoridad Nacional de Licencias Ambientales (Anla) decidió imponer a Hidroituango, uno de los proyectos hidroeléctricos más grandes del país, una multa que asciende a 753 millones de pesos colombianos debido a supuestos daños ambientales. Según lo reportado por Empresas Públicas de Medellín (EPM), la entidad operadora del proyecto, esta sanción se originó por presuntas irregularidades ambientales detectadas desde marzo de 2018, aunque la resolución formal que oficializa la multa data del pasado 17 de diciembre de 2025. Sin embargo, EPM solo recibió la notificación oficial el 30 de diciembre del mismo año, lo que ha generado un proceso de reacción interna y jurídica frente a la sanción impuesta. La serie de presuntas infracciones ambientales: captar volúmenes de agua por encima de los niveles otorgados en las concesiones, no realizar las acciones de reconformación y recuperación del cauce del río San Andrés ni de su zona de inundación, superar los límites máximos permitidos de material particulado en suspensión (PST) y sustancias contaminantes en la planta de asfalto de “El Valle”, así como incumplir medidas estipuladas en su plan de manejo y disposición de materiales y lugares de almacenamiento.
- La negociación entre la Electrificadora de Santander, ESSA, y el sindicato Sintraelecol: Mientras que los trabajadores buscan incrementos salariales y de beneficios similares al aumento del salario mínimo (23.7 %), la compañía plantea que el alza sea algunos puntos por encima del Índice de Precios al Consumidor, IPC, que bordea el 5.3 %. La brecha entre ambos valores es cercana a los 18 puntos porcentuales.
- Tras un cese de operaciones que se extendió por más de doce meses, Empresas Públicas de Medellín (EPM) oficializó la reactivación de la cadena de generación hidroeléctrica ubicada en el municipio de Sonsón, en el Oriente antioqueño. La puesta en marcha de estas centrales requirió una inversión aproximada de $24.000 millones, destinados a la rehabilitación integral de su infraestructura física y electromecánica. El proyecto de recuperación se centró en las Pequeñas Centrales Hidroeléctricas (PCH) Sonsón I y Sonsón II. Según los reportes técnicos de la compañía, la intervención incluyó el desmontaje y la sustitución de una tubería de carga de 875 metros de longitud en la central Sonsón I, además de diversas obras civiles y la modernización de los equipos de generación.
- Ecopetrol Emprende, el programa de fortalecimiento empresarial que en 2025 dejó 440 empleos directos e indirectos en Cartagena, fortaleciendo el tejido productivo en la zona de influencia de la refinería. El programa incluyó procesos de capacitación empresarial, asesoría especializada y la entrega de 1.739 elementos de dotación, entre maquinaria, equipos tecnológicos y soluciones digitales.
- El precio del título de Ecopetrol fue el más alto de la jornada (5 a 6 de enero de 2026) de las acciones negociadas en la Bolsa de Valores de Colombia, BVC. La petrolera tuvo un repunte de 5,21%, con un precio de $2.020. Los analistas estiman que este repunte se debe principalmente por el aumento del precio del Brent, el cual avanzó alrededor de 1.68% hoy, debido a que la Opep decidió en su reunión del 5 de enero no aumentar su producción, lo cual ayuda a mantener el precio del crudo en sus niveles actuales.
- 2026 empezó con la noticia de que el Gobierno del presidente de Estados Unidos, Donald Trump, capturó a Nicolás Maduro como parte de una nueva doctrina Monroe dentro de Latinoamérica. Este movimiento tiene repercusiones sobre el mercado petrolero. Sobre el medio día de ayer 6 de enero, la acción de Ecopetrol creció 3,6% en su cotización en el mercado de Estados Unidos y llegó a un precio de hasta US$11,21. El buen comportamiento que ha tenido la acción de la petrolera en Wall Street, de acuerdo a expertos, es debido a las incursiones de Estados Unidos en Venezuela, lo que ha incrementado la especulación, por lo que los inversionistas internacionales que están haciendo apuestas por el mercado venezolano e iraní, ya que se cambiaría el orden de producción de petróleo, así como optimismo por la elecciónpresidencial de mayo de 2026 en Colombia con una nueva expectativa de un posible cambio de rumbo económico.
- Para el 7 de enero de 2026, Ecopetrol anunció por medio de un comunicado que acordó con JP Morgan, entidad depositaria de su programa de ADRs, American Depositary Receipts, la extensión del acuerdo del pasado 27 de junio de 2025. Dicho acuerdo busca que haya una reducción de hasta 50% del costo de la emisión de ADRs negociadas en la Bolsa de Nueva York.
- El 7 de enero de 2026, se conoció una presunta denuncia sobre que Ecopetrol habría frenado una licitación del Invías para la vía Puerto Gaitán–Rubiales y terminaron repartiendo $213.000 millones entre entidades locales, con contratos a empresas cercanas a políticos del Meta, posibles sobrecostos y falta de competencia, mientras la carretera —clave para el petróleo del país— sigue inconclusa, con solo ocho kilómetros pavimentados y un tramo sin recursos a corte de noviembre del año pasado.
"""
TERM_INDUSTRIA = {
    "tarifas energéticas",
    "subsidios de energía",
    "soberanía energética",
    "regasificación",
    "gas natural",
    "crisis del gas",
    "energía eléctrica",
}

TERM_ESTADO = {
    "Ecopetrol",
    "Ricardo Roa",
    "Edwin Palma",
    "Ministerio de Minas y Energía",
    "Ministro de Minas y Energía",
    "CREG",
    "Air-e",
    "Regasificación",
}

TERM_EMPRESAS = {
    "Enel Colombia",
    "EPM",
    "Afinia",
    "Cens",
    "Celsia",
    "Essa",
    "Compañía Energética de Occidente",
}

TERM_ENFRAGEN = {
    "EnfraGen",
    "Termovalle",
    "Termoflores",
}

def extraer_fechas(pregunta):
    pregunta = pregunta.lower()

    # Caso 1: rango tipo "del 25 al 29 de agosto"
    match = re.search(r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 2: rango tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 3: una sola fecha "el 27 de agosto"
    match = re.search(r"(\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)", pregunta)
    if match:
        fecha = dateparser.parse(match.group(), languages=['es'])
        return fecha.date(), fecha.date()

    # Caso 4: sin fecha → None, None
    return None, None

# 2️⃣ Obtener fecha más reciente disponible
def obtener_fecha_mas_reciente(df):
    fecha_max = df["Fecha"].max()
    # Si es pandas Timestamp (tiene método .date), conviértelo
    if hasattr(fecha_max, "date"):
        return fecha_max.date()
    # Si ya es datetime.date, devuélvelo directo
    return fecha_max


# 3️⃣ Detectar sentimiento deseado
def detectar_sentimiento_deseado(pregunta):
    pregunta = pregunta.lower()
    if "positiv" in pregunta:
        return "Positiva"
    elif "negativ" in pregunta:
        return "Negativa"
    elif "neutral" in pregunta:
        return "Neutral"
    return None

# 4️⃣ Extraer entidades (personajes/instituciones/empresas/temas) - versión EnfraGen
def extraer_entidades(texto):
    t = (texto or "").lower()

    # ✅ Personajes / cargos
    personajes = {
        "Ricardo Roa": ["roa", "ricardo roa"],
        "Edwin Palma": ["palma", "edwin palma", "ministro edwin palma"],
        "Gustavo Petro": ["presidente petro", "gustavo petro"],
    }

    # ✅ Instituciones / entes / empresas relevantes para preguntas
    entidades = {
        "Ecopetrol": ["ecopetrol"],
        "Ministerio de Minas y Energía": ["minminas", "ministerio de minas", "ministerio de minas y energía", "minas y energía"],
        "CREG": ["creg", "comisión de regulación de energía y gas"],
        "DIAN": ["dian"],
        "CNE": ["cne", "consejo nacional electoral"],
        "Procuraduría": ["procuraduria", "procuraduría"],
        "Fiscalía": ["fiscalia", "fiscalía"],
        "Contraloría": ["contraloria", "contraloría"],
        "Air-e": ["air-e", "air e"],
        "EnfraGen": ["enfragen", "enfragen colombia"],
        "Termovalle": ["termovalle"],
        "Termoflores": ["termoflores"],
        "Enel Colombia": ["enel", "enel colombia"],
        "EPM": ["epm"],
        "Afinia": ["afinia"],
        "Cens": ["cens"],
        "Celsia": ["celsia"],
        "Essa": ["essa"],
        "Compañía Energética de Occidente": ["compañía energética de occidente", "compañía energética de occidente", "ceo"],
        "Sirius": ["sirius"],
        "Coveñas": ["coveñas", "covenas"],
        "Reficar": ["reficar"],
        "Permian": ["permian", "bloque permian"],
        "Oxy": ["oxy", "occidental", "occidental petroleum"],
    }

    # ✅ “Temas” (los términos del scraper) detectados desde la pregunta
    # Nota: esto NO filtra por sí solo; solo etiqueta para que lo uses en lógica/prompt si luego quieres.
    temas = {
        "tarifas energéticas": ["tarifas", "tarifarias", "tarifa", "factura de energía", "factura de luz"],
        "subsidios de energía": ["subsidios", "opción tarifaria", "opcion tarifaria", "rezago", "deuda", "diferimientos"],
        "soberanía energética": ["soberanía energética", "soberania energetica"],
        "regasificación": ["regasificación", "regasificacion", "gnl", "gas natural licuado"],
        "gas natural": ["gas natural"],
        "crisis del gas": ["crisis del gas", "escasez de gas", "desabastecimiento", "déficit de gas", "deficit de gas"],
        "energía eléctrica": ["energía eléctrica", "energia electrica", "electricidad", "generación", "generacion"],
    }

    encontrados = {
        "personajes": [],
        "entidades": [],
        "temas": [],
        "lugares": [],      # lo dejo para compatibilidad con tu filtro actual
        "categorias": [],   # idem
    }

    for nombre, sinonimos in personajes.items():
        if any(s in t for s in sinonimos):
            encontrados["personajes"].append(nombre)

    for nombre, sinonimos in entidades.items():
        if any(s in t for s in sinonimos):
            encontrados["entidades"].append(nombre)

    for nombre, sinonimos in temas.items():
        if any(s in t for s in sinonimos):
            encontrados["temas"].append(nombre)

    return encontrados


# 5️⃣ Filtrar titulares por entidades/temas y sentimiento (EnfraGen)
def filtrar_titulares(df_filtrado, entidades, sentimiento_deseado):
    """
    Filtra titulares usando lo detectado en la pregunta:
    - personajes y entidades: se buscan en el Título
    - temas: se filtran por match exacto contra la columna 'Término' (tal cual viene en el CSV)
    - sentimiento: filtro opcional (señal ligera)
    """
    if df_filtrado is None or df_filtrado.empty:
        return pd.DataFrame()

    filtro = df_filtrado.copy()
    condiciones = []

    # Normalizar columna Título por seguridad
    titulo_lower = filtro["Título"].astype(str).str.lower()

    # 1) Personajes (en Título)
    pers = (entidades or {}).get("personajes", []) or []
    if pers:
        condiciones.append(
            titulo_lower.str.contains("|".join([re.escape(p.lower()) for p in pers]), na=False)
        )

    # 2) Entidades / instituciones / empresas (en Título)
    ents = (entidades or {}).get("entidades", []) or []
    if ents:
        condiciones.append(
            titulo_lower.str.contains("|".join([re.escape(e.lower()) for e in ents]), na=False)
        )

    # 3) Temas (match exacto vs columna 'Término')
    temas = (entidades or {}).get("temas", []) or []
    if temas and "Término" in filtro.columns:
        termino_lower = filtro["Término"].astype(str).str.strip().str.lower()
        condiciones.append(termino_lower.isin([t.lower() for t in temas]))

    # Si hubo condiciones → OR entre todas
    if condiciones:
        filtro = filtro[pd.concat(condiciones, axis=1).any(axis=1)]

    # 4) Sentimiento (señal ligera)
    if sentimiento_deseado and "Sentimiento" in filtro.columns:
        filtro = filtro[filtro["Sentimiento"] == sentimiento_deseado]

    return filtro


# 7️⃣ Nube de palabras con colores y stopwords personalizadas
import random

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colores = [
        "rgb(64, 90, 104)",  # gris
        "rgb(0, 179, 115)",   # verde
        "rgb(240, 90, 47)"   # rojo
    ]
    return random.choice(colores)


def generar_nube(titulos, archivo_salida):
    texto = " ".join(titulos)
    texto = re.sub(r"[\n\r]", " ", texto)
    stopwords = set([
        "dice", "tras", "pide", "va", "día",
        "van", "ser", "hoy", "año", "años", "nuevo", "nueva", "será",
        "sobre", "entre", "hasta", "donde", "desde", "como", "pero", "también", "porque", "cuando",
        "ya", "con", "sin", "del", "los", "las", "que", "una", "por", "para", "este", "esta", "estos",
        "estas", "tiene", "tener", "fue", "fueron", "hay", "han", "son", "quien", "quienes", "le",
        "se", "su", "sus", "lo", "al", "el", "en", "y", "a", "de", "un", "es", "si", "quieren", "aún",
        "mantiene", "buscaría", "la", "haciendo", "recurriría", "ante", "meses", "están", "subir",
        "ayer", "prácticamente", "sustancialmente", "busca", "cómo", "qué", "días", "construcción","tariffs",
        "aranceles","construcción", "así", "no","irá", "está", "sea", "eso", "Ecopetrol"
    ])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        color_func=color_func,
        collocations=False,
        max_words=10
    ).generate(texto)
    wc.to_file(archivo_salida)


def generar_resumen_y_datos(fecha_str):
    """
    Genera el resumen diario, la nube de palabras y la selección de titulares,
    ahora con estructura temática obligatoria en hasta 4 párrafos:

    Mantiene:
    - Cache en /resumenes/resumen_{fecha}.txt
    - resumenes_metadata.csv
    - resumenes_index.faiss
    - Subida a S3
    - Nube de palabras
    - Lista de titulares del día
    """
    # Normalizar fecha y filtrar noticias del día
    fecha_dt = pd.to_datetime(fecha_str, errors="coerce").date()
    noticias_dia = df[df["Fecha"].dt.date == fecha_dt]
    
    # Helpers (EnfraGen)
    def _norm_term(x):
        return str(x).strip()

    def _top_titulares_por_bloque(df_bloque, max_items=18):
        """
        Devuelve una lista de strings con titulares priorizados por frecuencia (título repetido).
        """
        if df_bloque is None or df_bloque.empty:
            return []

        vc = df_bloque["Título"].astype(str).value_counts()
        top_titles = list(vc.index[:max_items])

        lineas = []
        usados = set()
        for titulo in top_titles:
            filas = df_bloque[df_bloque["Título"].astype(str) == titulo]
            row = filas.iloc[0]
            medio = str(row.get("Fuente", "")).strip()
            enlace = str(row.get("Enlace", "")).strip()
            key = (titulo, medio, enlace)
            if key in usados:
                continue
            usados.add(key)
            if medio:
                lineas.append(f"- {titulo} ({medio}) {enlace}".strip())
            else:
                lineas.append(f"- {titulo} {enlace}".strip())

        return lineas[:max_items]

    # =============================================================================
    # FIRMA DEL DATASET DEL DÍA (para detectar cambios en las noticias)
    # =============================================================================
    import hashlib

    titulos_dia = (
        noticias_dia["Título"]
        .fillna("")
        .str.strip()
        .sort_values()
        .tolist()
    )

    firma_str = "||".join(titulos_dia)
    firma_dataset = hashlib.md5(firma_str.encode("utf-8")).hexdigest()

    if noticias_dia.empty:
        return {"error": f"No hay noticias para la fecha {fecha_str}"}
    # =============================================================================
    # 🔥 PREPARAR TITULARES + NUBE DESDE EL INICIO (para que existan antes de guardar metadata)
    # =============================================================================
    os.makedirs("nubes", exist_ok=True)
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)

    # ✅ Titulares del día priorizados por frecuencia (máx 8)
    vc = noticias_dia["Título"].astype(str).value_counts()

    titulares_info = []
    for _, row in noticias_dia.iterrows():
        titulares_info.append({
            "titulo": row.get("Título", ""),
            "medio": row.get("Fuente", ""),
            "enlace": row.get("Enlace", "")
        })

    # ✅ Nube del día: con TODOS los títulos del día
    generar_nube(noticias_dia["Título"].fillna("").astype(str).tolist(), archivo_nube_path)



    # =============================================================================
    # 1️⃣ BLOQUES ENFRAGEN POR TÉRMINO (match exacto) + CONTEXTOS
    # =============================================================================
    noticias_dia = noticias_dia.copy()
    if "Término" in noticias_dia.columns:
        noticias_dia["Término"] = noticias_dia["Término"].apply(_norm_term)
    else:
        noticias_dia["Término"] = ""

    df_ind = noticias_dia[noticias_dia["Término"].isin(TERM_INDUSTRIA)]
    df_est = noticias_dia[noticias_dia["Término"].isin(TERM_ESTADO)]
    df_emp = noticias_dia[noticias_dia["Término"].isin(TERM_EMPRESAS)]
    df_enf = noticias_dia[noticias_dia["Término"].isin(TERM_ENFRAGEN)]

    ctx_ind = "\n".join(_top_titulares_por_bloque(df_ind, max_items=18))
    ctx_est = "\n".join(_top_titulares_por_bloque(df_est, max_items=18))
    ctx_emp = "\n".join(_top_titulares_por_bloque(df_emp, max_items=18))
    ctx_enf = "\n".join(_top_titulares_por_bloque(df_enf, max_items=18))

    ctx_general = "\n".join(_top_titulares_por_bloque(noticias_dia, max_items=24))


    # =============================================================================
    # 3️⃣ CONTEXTO NARRATIVO PREVIO (sólo días ANTERIORES a la fecha del resumen)
    # =============================================================================
    CONTEXTO_ANTERIOR = ""
    try:
        meta_path = "faiss_index/resumenes_metadata.csv"
        if os.path.exists(meta_path):
            df_prev = pd.read_csv(meta_path)

            if len(df_prev) > 0 and "fecha" in df_prev.columns:
                # Normalizar fechas a tipo date
                df_prev["fecha"] = pd.to_datetime(
                    df_prev["fecha"], errors="coerce"
                ).dt.date

                # Quedarnos SOLO con resúmenes de días anteriores al que vamos a resumir
                df_prev_anteriores = df_prev[df_prev["fecha"] < fecha_dt].sort_values("fecha")

                if len(df_prev_anteriores) > 0:
                    ultimos = df_prev_anteriores.tail(1)
                    contexto_texto = "\n\n".join(
                        f"({row['fecha']}) {str(row['resumen']).strip()}"
                        for _, row in ultimos.iterrows()
                    )

                    CONTEXTO_ANTERIOR = (
                        "CONTEXTO DEL ÚLTIMO DÍA ANTERIOR REGISTRADO:\n"
                        f"{contexto_texto}\n"
                    )

                    print(
                        f"🔗 Contexto narrativo cargado "
                        f"(último día anterior: {ultimos.iloc[-1]['fecha']})"
                    )
    except Exception as e:
        print(f"⚠️ No se pudo cargar el contexto narrativo: {e}")


    # =============================================================================
    # 4️⃣ PROMPT 
    # =============================================================================
    prompt = f"""
{CONTEXTO_ANTERIOR}

{CONTEXTO_POLITICO}

INSTRUCCIONES OBLIGATORIAS — LÉELAS TODAS ANTES DE ESCRIBIR

ROL
Eres un redactor técnico que elabora un BRIEF FACTUAL INTERNO.
NO eres analista, NO eres columnista, NO haces interpretación ni contexto adicional.

REGLAS FUNDAMENTALES (PROHIBICIONES ABSOLUTAS)
- Está TERMINANTEMENTE PROHIBIDO:
  - Introducir el texto con frases generales como:
    “Las noticias del día…”, “Las noticias de {fecha_str}…”, “Este día fue relevante…”
  - Explicar por qué algo es importante, relevante, significativo o preocupante.
  - Usar frases como:
    “lo que implica”, “lo que refuerza”, “lo que podría”, “lo que resalta”, “esto es clave”, “esto podría ser”.
  - Hacer inferencias, conclusiones, evaluaciones o lecturas políticas.
  - Agregar contexto que NO esté explícitamente contenido en los titulares o que no esté dentro de {CONTEXTO_POLITICO}.

  QUÉ SÍ PUEDES HACER
- Limitarte estrictamente a TRANSCRIBIR DE FORMA SINTÉTICA lo que dicen los titulares.
- Reescribir los hechos en prosa clara y neutra, sin calificarlos.
- Usar únicamente información que esté explícita en los titulares listados.
- Usar únicamente contexto que esté dentro de {CONTEXTO_POLITICO}

ESTILO
- Lenguaje neutro, seco y factual.
- Cada párrafo debe comenzar DIRECTAMENTE con el actor o el hecho (ej. “Ecopetrol…”, “Air-e…”).
- NO expliques consecuencias.
- NO relaciones hechos entre sí si los titulares no lo hacen explícitamente.
- SÍ puedes agregar frases cortas de contexto SOLO si ese dato está explícitamente en {CONTEXTO_POLITICO} y sirve para entender el titular o desarrollarlo mejor (desambiguar actor, rol institucional, estado de intervención, naturaleza pública/privada, o marco regulatorio inmediato).
Está prohibido usar ese contexto para inferir consecuencias, evaluar, o decir por qué importa.

USO PERMITIDO DEL {CONTEXTO_POLITICO} (SIN BARRERAS, PERO CONTROLADO)
- Puedes insertar micro-contexto (máx. 1 frase por párrafo) tomado de {CONTEXTO_POLITICO} cuando aporte claridad inmediata.
- Ese micro-contexto debe escribirse como HECHO, no como interpretación.
- Formato recomendado: una oración corta entre comas o como segunda oración.
  Ejemplos válidos:
  “La compañía informó que el gas provendrá de los campos Cupiagua, Cupiagua Sur y Pauto Sur, ubicados en el Piedemonte Llanero”, que es contexto del titular que dice que Ecopetrol ofrecerá en venta entre 39 y 68 millones de barriles diarios de gas natural, de enero a mayo de 2026.
Ejemplos prohibidos:
  “Esto refuerza…”, “esto implica…”, “esto es clave…”, “podría provocar…”


ESTRUCTURA OBLIGATORIA 
- Párrafo 1: INDUSTRIA (tarifas, subsidios, gas, regasificación, energía eléctrica) SOLO si hay titulares en ese bloque. Si no hay noticias sobre eso, elimina este párrafo.
- Párrafo 2: ESTADO + ENERGÍA (MinMinas, CREG, Ecopetrol, Air-e, funcionarios) SOLO si hay titulares en ese bloque. Si no hay noticias sobre eso, elimina este párrafo.
- Párrafo 3: EMPRESAS DEL SECTOR (Enel, EPM, Afinia, etc.) SOLO si hay titulares en ese bloque. Si no hay noticias sobre eso, elimina este párrafo.
- Párrafo 4: SOLO si hay titulares sobre EnfraGen / Termovalle / Termoflores. Si no hay noticias sobre eso, elimina este párrafo.

FORMATO
- Texto corrido.
- Separar párrafos únicamente con saltos de línea.
- NO usar títulos, encabezados ni etiquetas.
- NO usar listas ni viñetas.
- NO usar Markdown.
- Extensión: solo lo necesario para cubrir los hechos; si hay pocos titulares, el texto debe ser corto.


BLOQUE INDUSTRIA:
{ctx_ind}

BLOQUE ESTADO + ENERGÍA:
{ctx_est}

BLOQUE EMPRESAS DEL SECTOR:
{ctx_emp}

BLOQUE ENFRAGEN (solo si existe):
{ctx_enf}

CONTEXTO GENERAL DEL DÍA (referencia adicional):
{ctx_general}
"""


    # =============================================================================
    # 5️⃣ CACHE DE RESUMEN EN /resumenes
    # =============================================================================
    os.makedirs("resumenes", exist_ok=True)
    archivo_resumen = os.path.join(
        "resumenes",
        f"resumen_{fecha_str}.txt"
    )

    # -----------------------------------------------------------------------------
    # Archivo de firma del resumen (control de cambios del dataset)
    # -----------------------------------------------------------------------------
    archivo_firma = os.path.join(
        "resumenes",
        f"resumen_{fecha_str}_firma.txt"
        )
    # DECISIÓN: ¿REUTILIZAR RESUMEN O REHACER TODO?
    # =============================================================================

    rehacer_resumen = True

    if os.path.exists(archivo_resumen) and os.path.exists(archivo_firma):
        with open(archivo_firma, "r", encoding="utf-8") as f:
            firma_guardada = f.read().strip()

        if firma_guardada == firma_dataset:
            rehacer_resumen = False


    if not rehacer_resumen:
        # -------------------------------------------------------------------------
        # USAR RESUMEN EXISTENTE (dataset no cambió)
        # -------------------------------------------------------------------------
        with open(archivo_resumen, "r", encoding="utf-8") as f:
            resumen_texto = f.read()

    else:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista experto en ENERGÍA, con énfasis en el contexto colombiano."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=900
        )

        resumen_texto = respuesta.choices[0].message.content.strip()

        # Guardar resumen nuevo
        with open(archivo_resumen, "w", encoding="utf-8") as f:
            f.write(resumen_texto)

        # Guardar firma del dataset
        with open(archivo_firma, "w", encoding="utf-8") as f:
            f.write(firma_dataset)
        # =============================================================================
        # 9️⃣ EMBEDDINGS ACUMULATIVOS PARA RESÚMENES (FAISS)
        # =============================================================================
        try:
            os.makedirs("faiss_index", exist_ok=True)
            index_path = "faiss_index/resumenes_index.faiss"

            # Generar embedding del resumen del día
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=resumen_texto.strip()
            ).data[0].embedding
            emb_np = np.array([emb], dtype="float32")

            # Si el índice ya existe, cargarlo y agregar nuevo vector
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                index.add(emb_np)
                print(f"🧩 Embedding agregado al índice existente ({index.ntotal} vectores totales)")
            else:
                dim = len(emb_np[0])
                index = faiss.IndexFlatL2(dim)
                index.add(emb_np)
                print("🆕 Índice FAISS de resúmenes creado")

            faiss.write_index(index, index_path)
            print("💾 Guardado resumenes_index.faiss actualizado")

            r2_upload("resumenes_index.faiss")
            print("☁️ Subido resumenes_index.faiss a S3")

        except Exception as e:
            print(f"⚠️ Error al actualizar embeddings de resúmenes: {e}")
        # =============================================================================
        # 8️⃣ GUARDAR / ACTUALIZAR resumenes_metadata.csv Y SUBIR A S3
        # =============================================================================
        try:
            os.makedirs("faiss_index", exist_ok=True)
            resumen_meta_path = "faiss_index/resumenes_metadata.csv"

            df_resumen = pd.DataFrame([{
                "fecha": str(fecha_dt),
                "archivo_txt": f"resumen_{fecha_str}.txt",
                "nube": archivo_nube,
                "titulares": len(titulares_info),
                "resumen": resumen_texto.strip()
            }])

            # Si ya existe el archivo, lo leemos y agregamos (sin duplicar fechas)
            if os.path.exists(resumen_meta_path):
                df_prev = pd.read_csv(resumen_meta_path)
            else:
                df_prev = pd.DataFrame(columns=["fecha", "archivo_txt", "nube", "titulares", "resumen"])

            if str(fecha_dt) not in df_prev["fecha"].astype(str).values:
                df_total = pd.concat([df_prev, df_resumen], ignore_index=True)
                print(f"🆕 Agregado nuevo resumen para {fecha_dt}")
            else:
                print(f"♻️ Reemplazando resumen existente para {fecha_dt}")
                df_resumen = df_resumen.reindex(columns=df_prev.columns)
                df_prev.loc[df_prev["fecha"].astype(str) == str(fecha_dt), df_prev.columns] = df_resumen.values[0]
                df_total = df_prev

            df_total.to_csv(resumen_meta_path, index=False, encoding="utf-8")
            print(f"💾 Guardado local de resumenes_metadata.csv con {len(df_total)} fila(s) totales")
            r2_upload("resumenes_metadata.csv")
            print("☁️ Subido resumenes_metadata.csv a S3")
        except Exception as e:
            print(f"⚠️ No se pudo guardar/subir resumenes_metadata.csv: {e}")
    # =============================================================================
    # 6️⃣ SELECCIÓN DE TITULARES ALINEADOS A LOS 5 BLOQUES DEL RESUMEN
    # =============================================================================
    def limpiar(texto):
        return re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚüÜ0-9 ]", "", str(texto).lower())

    resumen_limpio = limpiar(resumen_texto)
        
    return {
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "titulares": titulares_info,
    }

@app.route("/resumen", methods=["POST"])
def resumen():
    print("🛰️ Solicitud recibida en /resumen")
    data = request.get_json()
    print(f"📩 JSON recibido: {data}")
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    resultado = generar_resumen_y_datos(fecha_str)

    if "error" in resultado:
        return jsonify(resultado), 404

    # 🧹 Evitar NaN en la respuesta
    import math
    resultado = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in resultado.items()} if isinstance(resultado, dict) else resultado

    return jsonify(_json_sanitize(resultado))

def extraer_rango_fechas(pregunta):
    # Busca expresiones tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre el (\d{1,2}) y el (\d{1,2}) de ([a-zA-Z]+)(?: de (\d{4}))?", pregunta.lower())
    if match:
        dia_inicio, dia_fin, mes, anio = match.groups()
        anio = anio if anio else str(datetime.now().year)
        fecha_inicio = dateparser.parse(f"{dia_inicio} de {mes} de {anio}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} de {mes} de {anio}", languages=['es'])
        if fecha_inicio and fecha_fin:
            return fecha_inicio.date(), fecha_fin.date()
    return None, None

# -----------------------------------------
# 🆕 Helper para obtener semanas reales del mes (lunes–viernes)
# -----------------------------------------
def normalizar_frase_semanas(texto: str) -> str:
    """
    Normaliza frases del tipo:
    - 'entre la primera semana de noviembre y la segunda?'
    - 'entre la primera semana de noviembre y la segunda de noviembre?'

    para que queden como:
    - 'entre la primera semana de noviembre y la segunda semana de noviembre'
    """

    meses_regex = (
        r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|setiembre|octubre|noviembre|diciembre"
    )

    # ¿Hay alguna referencia explícita a 'X semana de <mes>'?
    m = re.search(
        r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+(" + meses_regex + r")",
        texto,
        re.IGNORECASE,
    )
    if not m:
        return texto  # si no hay semanas del mes, no tocamos nada

    mes = m.group(2)

    # 1) Caso: '... y la segunda de noviembre' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\s+de\s+" + mes + r"\b",
        lambda m3: f" y la {m3.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # 2) Caso: '... y la segunda?' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\b(?!\s+semana)",
        lambda m2: f" y la {m2.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # Limpieza de espacios dobles
    texto = re.sub(r"\s{2,}", " ", texto)

    return texto

def obtener_semanas_del_mes(anio, mes, fecha_min_dataset, fecha_max_dataset):
    """
    Devuelve una lista de rangos semanales reales dentro de un mes:
    - Cada semana inicia en LUNES
    - Cada semana termina en DOMINGO, pero luego se ajusta al dataset
    - Solo se devuelven semanas que tengan algún día dentro del dataset
    """
    semanas = []

    # Primer día del mes
    desde = datetime(anio, mes, 1).date()

    # Último día del mes
    if mes == 12:
        hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
    else:
        hasta = datetime(anio, mes + 1, 1).date() - timedelta(days=1)

    # Mover "desde" al lunes de esa semana
    inicio = desde - timedelta(days=desde.weekday())  # weekday: lunes=0

    while inicio <= hasta:
        fin = inicio + timedelta(days=6)

        # Ajustar al mes
        real_inicio = max(inicio, desde)
        real_fin = min(fin, hasta)

        # Ajustar al dataset
        final_inicio = max(real_inicio, fecha_min_dataset)
        final_fin = min(real_fin, fecha_max_dataset)

        # Si el rango tiene al menos un día válido → agregarlo
        if final_inicio <= final_fin:
            semanas.append((final_inicio, final_fin))

        # Siguiente semana
        inicio += timedelta(days=7)

    return semanas

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

def interpretar_rango_fechas(pregunta: str, df_noticias: pd.DataFrame):
    """
    Interpreta fechas o rangos mencionados en la pregunta y los ajusta
    al rango disponible en df_noticias.

    Devuelve (fecha_inicio, fecha_fin, origen), donde las fechas son date o None.
    """
    if df_noticias is None or df_noticias.empty:
        return None, None, "sin_datos"

    fechas_validas = df_noticias["Fecha"].dropna()
    if fechas_validas.empty:
        return None, None, "sin_datos"

    fecha_min = fechas_validas.min().date()
    fecha_max = fechas_validas.max().date()

    texto = (pregunta or "")
    texto_lower = texto.lower()
    texto_lower = normalizar_frase_semanas(texto_lower)

    fecha_inicio = None
    fecha_fin = None
    origen = "sin_fecha"

    # 1️⃣ Casos relativos: "esta semana", "hoy", "ayer"
    if fecha_inicio is None and fecha_fin is None:
        if "esta semana" in texto_lower:
            fecha_fin = fecha_max
            fecha_inicio = max(fecha_min, fecha_max - timedelta(days=6))
            origen = "esta_semana_dataset"
        elif re.search(r"\bhoy\b", texto_lower):
            fecha_inicio = fecha_fin = fecha_max
            origen = "hoy_dataset"
        elif re.search(r"\bayer\b(?=[\s,.!?;:]|$)", texto_lower):
            candidata = fecha_max - timedelta(days=1)
            if candidata < fecha_min:
                candidata = fecha_min
            fecha_inicio = fecha_fin = candidata
            origen = "ayer_dataset"

    # 2️⃣ Rango de semanas:
    #    - "entre la primera y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        # Forma 1: entre la primera y la segunda semana de noviembre
        patron1 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+y\s+"
            r"(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        # Forma 2: entre la primera semana de noviembre y la segunda semana de noviembre
        patron2 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+\2",
            texto_lower,
        )
        # Forma 3: entre la primera semana de noviembre y la segunda de noviembre
        patron3 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+de\s+\2",
            texto_lower,
        )
        if patron1 or patron2 or patron3:
            if patron1:
                ord1, ord2, nombre_mes = patron1.groups()
            elif patron2:
                # patron2: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron2.groups()
            else:
                # patron3: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron3.groups()

            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year
                # Inicio y fin del mes calendario
                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                # Semanas "fijas" (1–7, 8–14, 15–21, 22–28)
                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                ordenes = ["primera", "segunda", "tercera", "cuarta"]
                i1 = ordenes.index(ord1)
                i2 = ordenes.index(ord2)
                idx_min, idx_max = min(i1, i2), max(i1, i2)

                fecha_inicio, _ = semanas[idx_min]
                _, fecha_fin = semanas[idx_max]
                origen = "rango_semanas_mes"



    # 3️⃣ Una sola semana: "primera/segunda/tercera/cuarta semana de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_semana_mes = re.search(
            r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower
        )
        if m_semana_mes:
            ord_semana = m_semana_mes.group(1)
            nombre_mes = m_semana_mes.group(2)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                idx = ["primera", "segunda", "tercera", "cuarta"].index(ord_semana)
                fecha_inicio, fecha_fin = semanas[idx]
                origen = "semana_del_mes"
    # 3.5️⃣ Rango por meses: "entre septiembre y diciembre" / "de septiembre a diciembre"
    if fecha_inicio is None and fecha_fin is None:
        m_rango_meses = re.search(
            r"(?:entre|de)\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+"
            r"(?:y|a|hasta)\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower
        )

        if m_rango_meses:
            mes1 = MESES_ES.get(m_rango_meses.group(1))
            mes2 = MESES_ES.get(m_rango_meses.group(2))

            # año: si viene explícito en la pregunta úsalo; si no, usa el año del dataset
            m_anio = re.search(r"(20\d{2})", texto_lower)
            anio = int(m_anio.group(1)) if m_anio else fecha_max.year

            if mes1 and mes2:
                # Si el rango cruza el año (ej: entre noviembre y febrero)
                anio_fin = anio if mes2 >= mes1 else anio + 1

                desde = datetime(anio, mes1, 1).date()

                # último día del mes final
                if mes2 == 12:
                    hasta = datetime(anio_fin + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio_fin, mes2 + 1, 1).date() - timedelta(days=1)

                fecha_inicio, fecha_fin = desde, hasta
                origen = "rango_meses"

    # 4️⃣ Mes completo: "en noviembre", "durante noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_mes = re.search(
            r"en\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        if m_mes:
            nombre_mes = m_mes.group(1)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                fecha_inicio, fecha_fin = desde, hasta
                origen = "mes_completo"

    # 5️⃣ Rangos explícitos "entre el 3 y el 7 de noviembre" / "del 3 al 7 de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        patron_entre = re.search(
            r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        patron_del = re.search(
            r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )

        m = patron_entre or patron_del
        if m:
            dia1 = int(m.group(1))
            dia2 = int(m.group(2))
            nombre_mes = m.group(3)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                d_ini = min(dia1, dia2)
                d_fin = max(dia1, dia2)

                try:
                    fecha_inicio = datetime(anio, mes_num, d_ini).date()
                    fecha_fin = datetime(anio, mes_num, d_fin).date()
                    origen = "rango_explicito_texto"
                except ValueError:
                    fecha_inicio = None
                    fecha_fin = None
                    origen = "sin_fecha_valida"

    # 6️⃣ Último intento con search_dates (fecha puntual o rango)
    if fecha_inicio is None and fecha_fin is None and "search_dates" in globals():
        try:
            resultados = search_dates(
                texto,
                languages=["es"],
                settings={"RELATIVE_BASE": datetime.combine(fecha_max, datetime.min.time())},
            ) or []
        except Exception:
            resultados = []

        if resultados:
            # Solo aceptamos fragmentos que tengan algún dígito
            # (evita que expresiones vagas como "últimas encuestas"
            # se tomen como una fecha puntual).
            resultados_filtrados = []
            for frag, fecha_dt in resultados:
                if re.search(r"\d", frag):
                    resultados_filtrados.append((frag, fecha_dt))
                                                
            if resultados_filtrados:
                fechas_detectadas = [r[1].date() for r in resultados_filtrados]
            else:
                fechas_detectadas = []
            if fechas_detectadas:
                if (
                    ("entre " in texto_lower or " del " in texto_lower or "del " in texto_lower or "desde " in texto_lower)
                    and len(fechas_detectadas) >= 2
                ):
                    fecha_inicio = min(fechas_detectadas[0], fechas_detectadas[1])
                    fecha_fin = max(fechas_detectadas[0], fechas_detectadas[1])
                    origen = "rango_explicito_search_dates"
                else:
                    fecha_inicio = fecha_fin = fechas_detectadas[0]
                    origen = "fecha_puntual"


    # 7️⃣ Ajustar al rango del dataset
    if fecha_inicio is not None and fecha_fin is not None:
        original_inicio, original_fin = fecha_inicio, fecha_fin
        fecha_inicio = max(fecha_inicio, fecha_min)
        fecha_fin = min(fecha_fin, fecha_max)

        if fecha_inicio > fecha_fin:
            return None, None, "fuera_rango_dataset"

        if (fecha_inicio, fecha_fin) != (original_inicio, original_fin):
            origen += "_ajustada_dataset"

    return fecha_inicio, fecha_fin, origen



def filtrar_docs_por_rango(docs, fecha_inicio, fecha_fin):
    """
    Filtra una lista de Document de LangChain por metadata['fecha'] en el rango dado.
    Devuelve (docs_filtrados, se_aplico_filtro: bool).
    """
    if not docs or not fecha_inicio or not fecha_fin:
        return docs, False

    filtrados = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        fecha_meta = meta.get("fecha")
        if not fecha_meta:
            continue
        try:
            f = pd.to_datetime(fecha_meta).date()
        except Exception:
            continue
        if fecha_inicio <= f <= fecha_fin:
            filtrados.append(d)

    if filtrados:
        return filtrados, True
    else:
        # Si el filtro deja todo vacío, devolvemos la lista original
        # para no quedarnos sin contexto.
        return docs, False
    
def construir_vectorstore_desde_df(df_rango: pd.DataFrame):
    """Crea un mini-vectorstore (LangChain FAISS) con noticias ya filtradas por fecha.
    Usa las mismas embeddings globales.
    """
    docs = []
    if df_rango is None or df_rango.empty:
        return None

    for _, row in df_rango.iterrows():
        titulo = str(row.get("Título", "")).strip()
        if not titulo:
            continue

        fecha = row.get("Fecha", "")
        try:
            fecha_str = pd.to_datetime(fecha, errors="coerce").date().isoformat()
        except Exception:
            fecha_str = str(fecha)[:10] if fecha else ""

        metadata = {
            "titulo": titulo,
            "fuente": str(row.get("Fuente", "")).strip(),
            "enlace": str(row.get("Enlace", "")).strip(),
            "fecha": fecha_str,
            "sentimiento": str(row.get("Sentimiento", "")).strip(),
            "termino": str(row.get("Término", "")).strip(),
        }

        docs.append(Document(page_content=titulo, metadata=metadata))

    if not docs:
        return None

    return LCFAISS.from_documents(docs, embeddings)

#pregunta!!!!    
# ------------------------------
# 🤖 Endpoint /pregunta (RAG con filtro por fecha antes de FAISS)
# ------------------------------
def detectar_comparacion_meses(texto):
    """
    Detecta preguntas tipo:
    - "compara septiembre vs diciembre"
    - "septiembre contra diciembre"
    - "diferencias entre septiembre y diciembre"
    Devuelve (mes1, mes2) como enteros o (None, None).
    """
    if not texto:
        return None, None

    t = texto.lower()

    # Caso: "compara X vs Y" / "diferencias entre X y Y"
    m = re.search(
        r"(?:compara|comparar|comparación|diferenc|diferencia|vs|contra|versus).*?\b"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b"
        r".*?\b(vs|contra|versus|y|a)\b.*?\b"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b",
        t
    )

    # Variante simple: "septiembre vs diciembre"
    if not m:
        m = re.search(
            r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b"
            r"\s+(?:vs|contra|versus)\s+"
            r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b",
            t
        )
        if not m:
            return None, None

        mes1 = MESES_ES.get(m.group(1))
        mes2 = MESES_ES.get(m.group(2))
        return mes1, mes2

    # En el primer regex, el segundo mes es group(3)
    mes1 = MESES_ES.get(m.group(1))
    mes2 = MESES_ES.get(m.group(3))
    return mes1, mes2

ORDINAL_SEMANA = {
    "primera": 1, "1ra": 1, "1": 1,
    "segunda": 2, "2da": 2, "2": 2,
    "tercera": 3, "3ra": 3, "3": 3,
    "cuarta": 4, "4ta": 4, "4": 4,
    "quinta": 5, "5ta": 5, "5": 5,
}

def inicio_fin_semana_iso(fecha_date):
    """Devuelve (lunes, domingo) de la semana ISO de fecha_date."""
    lunes = fecha_date - timedelta(days=fecha_date.weekday())
    domingo = lunes + timedelta(days=6)
    return lunes, domingo

def rango_semana_n_de_mes(anio, mes, n_semana):
    """
    Semana n del mes: 1=primer bloque lunes-domingo que cae en el mes.
    Devuelve (inicio, fin). Puede arrancar en mes anterior si el lunes cae antes.
    """
    first_day = datetime(anio, mes, 1).date()
    first_monday, first_sunday = inicio_fin_semana_iso(first_day)
    start = first_monday + timedelta(days=7*(n_semana-1))
    end = start + timedelta(days=6)
    return start, end

def detectar_comparacion_semanas(texto, fecha_max_dataset):
    """
    Detecta comparación semanal y devuelve (rangoA, rangoB, origen) o (None,None,None).

    Soporta MVP:
    1) "esta semana vs semana pasada/anterior"
    2) "primera/segunda/... semana de <mes> vs primera/segunda/... semana de <mes>"
       (si el segundo mes no aparece, asume el mismo)
    3) "semana del 4 al 10 de diciembre vs semana del 11 al 17 de diciembre"
    """
    if not texto:
        return None, None, None

    t = texto.lower().strip()

    # --- Caso 1: esta semana vs semana pasada/anterior
    if ("esta semana" in t or "la semana" in t) and ("semana pasada" in t or "semana anterior" in t):
        a_ini, a_fin = inicio_fin_semana_iso(fecha_max_dataset)
        b_ini = a_ini - timedelta(days=7)
        b_fin = a_fin - timedelta(days=7)
        # Por convención: A = esta semana, B = semana pasada
        return (a_ini, a_fin), (b_ini, b_fin), "comparacion_semanal_relativa"

    # Año explícito opcional
    m_anio = re.search(r"(20\d{2})", t)
    anio = int(m_anio.group(1)) if m_anio else fecha_max_dataset.year

    # --- Caso 2: "primera semana de diciembre vs segunda semana de diciembre"
# --- Caso 2: "primera semana de diciembre vs la segunda (semana) (de diciembre)"
    m = re.search(
        r"(?:la\s+)?(primera|segunda|tercera|cuarta|quinta|1ra|2da|3ra|4ta|5ta|1|2|3|4|5)\s+semana\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r".*?\b(vs|contra|versus|y)\b.*?"
        r"(?:la\s+)?(primera|segunda|tercera|cuarta|quinta|1ra|2da|3ra|4ta|5ta|1|2|3|4|5)"
        r"(?:\s+semana)?"
        r"(?:\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre))?",
        t
    )

    if m:
        ordA = ORDINAL_SEMANA.get(m.group(1))
        mesA = MESES_ES.get(m.group(2))
        ordB = ORDINAL_SEMANA.get(m.group(4))
        mesB = MESES_ES.get(m.group(5)) if m.group(5) else mesA  # si no dice mes, usa el mismo
        if ordA and mesA and ordB and mesB:
            rA = rango_semana_n_de_mes(anio, mesA, ordA)
            rB = rango_semana_n_de_mes(anio, mesB, ordB)
            return rA, rB, "comparacion_semanal_ordinal"

    # --- Caso 3: "semana del 4 al 10 de diciembre vs semana del 11 al 17 de diciembre"
    m2 = re.search(
        r"semana\s+del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r".*?\b(vs|contra|versus|y)\b.*?"
        r"semana\s+del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
        t
    )
    if m2:
        d1, d2, mes1 = int(m2.group(1)), int(m2.group(2)), MESES_ES.get(m2.group(3))
        d3, d4, mes2 = int(m2.group(5)), int(m2.group(6)), MESES_ES.get(m2.group(7))
        if mes1 and mes2:
            rA = (datetime(anio, mes1, d1).date(), datetime(anio, mes1, d2).date())
            rB = (datetime(anio, mes2, d3).date(), datetime(anio, mes2, d4).date())
            return rA, rB, "comparacion_semanal_explicita"

    return None, None, None


def es_pregunta_comparativa(texto):
    if not texto:
        return False
    t = texto.lower()
    # Señales claras de comparación
    return any(k in t for k in ["compara", "comparar", "vs", "versus", "contra", "diferencias", "diferente", "cambio", "cambió", "cambios"])
def metricas_periodo(dfX, max_tit=5):
    """
    Resumen estructurado para comparar periodos sin inventar.
    Prioriza frecuencia; sentimiento es señal ligera.
    """
    if dfX is None or dfX.empty:
        return {
            "total": 0,
            "top_terminos": {},
            "top_fuentes": {},
            "sentimiento": {},
            "titulares": []
        }

    top_terminos = {}
    if "Término" in dfX.columns:
        top_terminos = dfX["Término"].astype(str).value_counts().head(8).to_dict()

    top_fuentes = {}
    if "Fuente" in dfX.columns:
        top_fuentes = dfX["Fuente"].astype(str).value_counts().head(8).to_dict()

    sentimiento = {}
    if "Sentimiento" in dfX.columns:
        sentimiento = dfX["Sentimiento"].astype(str).value_counts().to_dict()

    # Titulares por frecuencia (evita duplicados)
    vc = dfX["Título"].astype(str).value_counts()
    titulares = []
    for titulo in list(vc.index[:max_tit]):
        row = dfX[dfX["Título"].astype(str) == titulo].iloc[0]
        titulares.append({
            "titulo": row.get("Título", ""),
            "medio": row.get("Fuente", ""),
            "enlace": row.get("Enlace", ""),
            "fecha": row.get("Fecha").strftime("%Y-%m-%d") if pd.notnull(row.get("Fecha")) else ""
        })

    return {
        "total": len(dfX),
        "top_terminos": top_terminos,
        "top_fuentes": top_fuentes,
        "sentimiento": sentimiento,
        "titulares": titulares
    }
def responder_pregunta(q: str):
    """Lógica central de /pregunta (reusable para frontend y Telegram).
    Devuelve (payload: dict, status_code: int).
    """
    q = (q or "").strip()
    if not q:
        return {"error": "No se proporcionó una pregunta válida."}, 400

    try:
        # 🧠 1️⃣ Detectar entidades y rango de fechas
        entidades = extraer_entidades(q) if "extraer_entidades" in globals() else {}

        # 🆚 1.A) Detectar comparación de meses (septiembre vs diciembre, etc.)
        mesA, mesB = detectar_comparacion_meses(q)

        modo_comparacion_meses = False
        rangoA = None
        rangoB = None

        if mesA and mesB:
            # Año: si viene explícito úsalo; si no, usa el año del dataset
            fechas_validas = df["Fecha"].dropna()
            anio_dataset = fechas_validas.max().year if not fechas_validas.empty else datetime.now().year

            m_anio = re.search(r"(20\d{2})", q.lower())
            anio = int(m_anio.group(1)) if m_anio else anio_dataset

            # Rango mes A
            desdeA = datetime(anio, mesA, 1).date()
            hastaA = (datetime(anio + (1 if mesA == 12 else 0), (mesA % 12) + 1, 1).date() - timedelta(days=1))

            # Rango mes B
            desdeB = datetime(anio, mesB, 1).date()
            hastaB = (datetime(anio + (1 if mesB == 12 else 0), (mesB % 12) + 1, 1).date() - timedelta(days=1))

            modo_comparacion_meses = True
            rangoA = (desdeA, hastaA)
            rangoB = (desdeB, hastaB)

            # Para que el resto del flujo NO caiga en "sin_fecha"
            fecha_inicio = min(desdeA, desdeB)
            fecha_fin = max(hastaA, hastaB)
        else:
            fecha_inicio, fecha_fin, origen_rango = interpretar_rango_fechas(q, df)

        print(f"📅 Rango interpretado para la pregunta: {fecha_inicio} → {fecha_fin} ({origen_rango if 'origen_rango' in locals() else 'mes_vs_mes'})")

        # 🧠 1.B) Detectar comparación semanal (usa la función REAL que sí existe en el archivo)
        fechas_validas = pd.to_datetime(df["Fecha"], errors="coerce").dropna()
        fecha_max_dataset = fechas_validas.max().date() if not fechas_validas.empty else datetime.now().date()

        rA, rB, origen_sem = detectar_comparacion_semanas(q, fecha_max_dataset)

        # ✅ Si hay comparación semanal detectada, ejecutamos modo comparación semanal
        if es_pregunta_comparativa(q) and rA and rB:
            (desdeA, hastaA) = rA
            (desdeB, hastaB) = rB

            df_validas = df.dropna(subset=["Fecha"]).copy()
            df_validas["Fecha_date"] = pd.to_datetime(df_validas["Fecha"], errors="coerce").dt.date

            # Recortar a rango real del dataset
            min_ds = df_validas["Fecha_date"].min()
            max_ds = df_validas["Fecha_date"].max()
            desdeA, hastaA = max(desdeA, min_ds), min(hastaA, max_ds)
            desdeB, hastaB = max(desdeB, min_ds), min(hastaB, max_ds)

            dfA = df_validas[(df_validas["Fecha_date"] >= desdeA) & (df_validas["Fecha_date"] <= hastaA)].copy()
            dfB = df_validas[(df_validas["Fecha_date"] >= desdeB) & (df_validas["Fecha_date"] <= hastaB)].copy()

            metA = metricas_periodo(dfA, max_tit=8)
            metB = metricas_periodo(dfB, max_tit=8)

            prompt_comp = f"""
Compara la cobertura mediática entre dos periodos de tiempo y responde SOLO con la información disponible.
No inventes hechos externos. Si falta información, dilo.

Pregunta del usuario:
{q}

Periodo A ({desdeA} a {hastaA}):
- Total de noticias: {metA['total']}
- Top términos: {metA['top_terminos']}
- Top fuentes: {metA['top_fuentes']}
- Sentimiento: {metA['sentimiento']}
- Titulares ejemplo: {[t['titulo'] for t in metA['titulares'][:5]]}

Periodo B ({desdeB} a {hastaB}):
- Total de noticias: {metB['total']}
- Top términos: {metB['top_terminos']}
- Top fuentes: {metB['top_fuentes']}
- Sentimiento: {metB['sentimiento']}
- Titulares ejemplo: {[t['titulo'] for t in metB['titulares'][:5]]}
"""

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Eres un analista del sector energético colombiano y de su cobertura mediática. Prohibido usar conocimiento externo."},
                    {"role": "user", "content": prompt_comp}
                ],
                temperature=0.2
            )

            respuesta = resp.choices[0].message.content.strip()
            respuesta = quitar_markdown_basico(respuesta)
            titulares_usados = metA["titulares"] + metB["titulares"][:MAX_TITULARES_SELECCION]

            return {
                "respuesta": respuesta,
                "titulares_usados": titulares_usados,
                "filtros": {
                    "entidades": entidades,
                    "rango": [str(desdeA), str(hastaA), str(desdeB), str(hastaB)],
                    "resumenes_usados": [],
                    "origen": origen_sem
                }
            }, 200

            prompt_comp = f"""
Compara la cobertura mediática entre dos periodos de tiempo (mes vs mes) y responde SOLO con la información disponible.
No inventes hechos externos. Si falta información, dilo.

Pregunta del usuario:
{q}

MES A ({desdeA} a {hastaA}):
- Total de noticias: {metA['total']}
- Top términos: {metA['top_terminos']}
- Top fuentes: {metA['top_fuentes']}
- Sentimiento: {metA['sentimiento']}
- Titulares ejemplo: {[t['titulo'] for t in metA['titulares'][:5]]}

MES B ({desdeB} a {hastaB}):
- Total de noticias: {metB['total']}
- Top términos: {metB['top_terminos']}
- Top fuentes: {metB['top_fuentes']}
- Sentimiento: {metB['sentimiento']}
- Titulares ejemplo: {[t['titulo'] for t in metB['titulares'][:5]]}
"""

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Eres un analista del sector energético colombiano y de su cobertura mediática. Prohibido usar conocimiento externo."},
                    {"role": "user", "content": prompt_comp}
                ],
                temperature=0.2
            )

            respuesta = resp.choices[0].message.content.strip()
            respuesta = quitar_markdown_basico(respuesta)
            titulares_usados = metA["titulares"] + metB["titulares"][:MAX_TITULARES_SELECCION]

            return {
                "respuesta": respuesta,
                "titulares_usados": titulares_usados,
                "filtros": {
                    "entidades": entidades,
                    "rango": [str(desdeA), str(hastaA), str(desdeB), str(hastaB)],
                    "resumenes_usados": [],
                    "origen": "comparacion_meses"
                }
            }, 200

        tiene_rango = fecha_inicio is not None and fecha_fin is not None

        # 🧠 2️⃣ Filtrar DataFrame por rango ANTES de FAISS (solo si hay rango)
        df_rango = pd.DataFrame()
        if tiene_rango:
            df_validas = df.dropna(subset=["Fecha"]).copy()
            df_validas["Fecha_date"] = pd.to_datetime(df_validas["Fecha"], errors="coerce").dt.date
            mask = (df_validas["Fecha_date"] >= fecha_inicio) & (df_validas["Fecha_date"] <= fecha_fin)
            df_rango = df_validas[mask].copy()
            print(f"🧾 Noticias en rango {fecha_inicio} → {fecha_fin}: {len(df_rango)} filas")

        # 🧠 3️⃣ Recuperar resúmenes relevantes (contexto macro)
        resumen_docs = []
        if retriever_resumenes is not None:
            try:
                resumen_docs = retriever_resumenes.invoke(q)
            except Exception as e:
                print(f"⚠️ Error al recuperar resúmenes con LangChain: {e}")
                resumen_docs = []
        else:
            print("⚠️ retriever_resumenes es None (aún no hay resúmenes indexados).")

        resumen_docs_filtrados, dias_resumen_usados = filtrar_docs_por_rango(resumen_docs, fecha_inicio, fecha_fin)

        # 🧠 4️⃣ Recuperar noticias relevantes (contexto micro)
        noticias_docs = []
        noticias_docs_filtrados = []

        if tiene_rango and (df_rango is not None) and (not df_rango.empty):
            try:
                vectorstore_rango = construir_vectorstore_desde_df(df_rango)
                retriever_rango = vectorstore_rango.as_retriever(search_kwargs={"k": 40})
                noticias_docs = retriever_rango.invoke(q)
                noticias_docs_filtrados = noticias_docs
            except Exception as e:
                print(f"⚠️ Error construyendo mini-vectorstore por rango: {e}")
                noticias_docs = []
                noticias_docs_filtrados = []
        else:
            if vectorstore_noticias is not None:
                try:
                    retriever_global = vectorstore_noticias.as_retriever(search_kwargs={"k": 40})
                    noticias_docs = retriever_global.invoke(q)
                except Exception as e:
                    print(f"⚠️ Error al recuperar noticias con vectorstore global: {e}")
                    noticias_docs = []
            else:
                print("⚠️ vectorstore_noticias es None (no se construyó índice global de noticias).")

            noticias_docs_filtrados = noticias_docs

        # 4.C) Si no hay NADA de contexto (ni resúmenes ni noticias), responde orientando
        if not resumen_docs_filtrados and not noticias_docs_filtrados:
            mensaje = (
                "No encontré noticias claramente relacionadas con tu pregunta en el histórico disponible. "
                "Intenta reformularla, por ejemplo:\n"
                "- Especifica un tema (aranceles, tasas de interés, nearshoring, etc.)\n"
                "- Menciona un país, ciudad o personaje.\n"
                "- Si quieres un periodo, indica las fechas aproximadas."
            )
            return {
                "respuesta": mensaje,
                "titulares_usados": [],
                "filtros": {
                    "entidades": entidades,
                    "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                    "resumenes_usados": [],
                }
            }, 200

        # 🧾 5️⃣ Construir bloque de titulares + lista para el frontend
        lineas_titulares = []
        titulares_usados = []
        vistos = set()

        for d in noticias_docs_filtrados:
            md = getattr(d, "metadata", {}) or {}
            titulo = md.get("titulo") or md.get("Título") or ""
            medio = md.get("fuente") or md.get("Fuente") or ""
            enlace = md.get("enlace") or md.get("Enlace") or ""
            fecha = md.get("fecha") or md.get("Fecha") or ""

            key = (titulo.strip(), medio.strip())
            if not titulo or key in vistos:
                continue
            vistos.add(key)

            lineas_titulares.append(f"- {titulo} ({medio}) [{fecha}]")

            titulares_usados.append({
                "titulo": titulo,
                "medio": medio,
                "enlace": enlace,
                "fecha": str(fecha)[:10] if fecha else ""
            })

            if len(titulares_usados) >= MAX_TITULARES_SELECCION:
                break

        bloque_titulares = "\n".join(lineas_titulares) if lineas_titulares else "No se encontraron titulares específicos, solo contexto general de resúmenes."

        # 🧾 6️⃣ Bloque de resúmenes para prompt
        lineas_resumenes = []
        for d in resumen_docs_filtrados[:6]:
            md = getattr(d, "metadata", {}) or {}
            fecha = md.get("fecha") or md.get("Fecha") or ""
            texto = getattr(d, "page_content", str(d)).strip()
            if texto:
                lineas_resumenes.append(f"[{fecha}] {texto}")

        bloque_resumenes = "\n\n".join(lineas_resumenes) if lineas_resumenes else "(No hay resúmenes relevantes en el periodo.)"

        texto_usuario = f"""{CONTEXTO_POLITICO}

Responde en español, con tono profesional y analítico, claro y directo.
Usa ÚNICAMENTE la información contenida en los “Resúmenes relevantes” y “Titulares relevantes”.
Está PROHIBIDO agregar contexto externo o hechos no presentes en esos bloques.
Si algo no aparece en titulares/resúmenes, dilo explícitamente.
REGLAS FUNDAMENTALES (PROHIBICIONES ABSOLUTAS)
- Está TERMINANTEMENTE PROHIBIDO:
  - Explicar por qué algo es importante, relevante, significativo o preocupante.
  - Usar frases como:
    “lo que implica”, “lo que refuerza”, “lo que podría”, “lo que resalta”, “esto es clave”, “esto podría ser”.
  - Hacer inferencias, conclusiones, evaluaciones o lecturas políticas.
  - Agregar contexto que NO esté explícitamente contenido en los titulares o que no esté dentro de {CONTEXTO_POLITICO}.
Reglas adicionales: 
  - NO expliques consecuencias.
  - NO relaciones hechos entre sí si los titulares no lo hacen explícitamente.
  - SÍ puedes agregar frases cortas de contexto SOLO si ese dato está explícitamente en {CONTEXTO_POLITICO} y sirve para entender el titular o desarrollarlo mejor (desambiguar actor, rol institucional, estado de intervención, naturaleza pública/privada, o marco regulatorio inmediato).
  - Está prohibido usar ese contexto para inferir consecuencias, evaluar, o decir por qué importa.

USO PERMITIDO DEL {CONTEXTO_POLITICO} (SIN BARRERAS, PERO CONTROLADO)
- Puedes insertar micro-contexto (máx. 1 frase por párrafo) tomado de {CONTEXTO_POLITICO} cuando aporte claridad inmediata.
- Ese micro-contexto debe escribirse como HECHO, no como interpretación.

Pregunta del usuario:
{q}

Rango temporal de referencia (si aplica):
{fecha_inicio} → {fecha_fin}

Resúmenes relevantes:
{bloque_resumenes}

Titulares relevantes:
{bloque_titulares}

Respuesta:
"""

        texto_respuesta = chain_pregunta.invoke({"texto_usuario": texto_usuario}).strip()

        return {
            "respuesta": texto_respuesta,
            "titulares_usados": titulares_usados,
            "filtros": {
                "entidades": entidades,
                "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                "resumenes_usados": dias_resumen_usados,
            }
        }, 200

    except Exception as e:
        print(f"❌ Error en /pregunta (LangChain): {e}")
        return {"error": str(e)}, 500

@app.route("/pregunta", methods=["POST"])
def pregunta():
    data = request.get_json(silent=True) or {}
    q = (data.get("pregunta", "") or "").strip()
    payload, status = responder_pregunta(q)
    return jsonify(payload), status

# =========================
# 🤖 TELEGRAM WEBHOOK
# =========================

import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_ENERGY", "").strip()
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
TELEGRAM_PUBLIC_URL = os.getenv("TELEGRAM_PUBLIC_URL", "").strip()

def telegram_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

def telegram_send_message(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ TELEGRAM_BOT_TOKEN no definido")
        return
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(telegram_api_url("sendMessage"), json=payload, timeout=20)
    except Exception as e:
        print("❌ Error enviando mensaje a Telegram:", e)

def set_telegram_webhook():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_PUBLIC_URL:
        print("⚠️ Telegram webhook no configurado (faltan variables)")
        return

    webhook_url = f"{TELEGRAM_PUBLIC_URL.rstrip('/')}/telegram_webhook"
    data = {"url": webhook_url}

    if TELEGRAM_WEBHOOK_SECRET:
        data["secret_token"] = TELEGRAM_WEBHOOK_SECRET

    try:
        r = requests.post(telegram_api_url("setWebhook"), data=data, timeout=20)
        print("📡 setWebhook status:", r.status_code, r.text[:300])
    except Exception as e:
        print("❌ Error seteando webhook:", e)

@app.route("/telegram_webhook", methods=["POST"])
def telegram_webhook():
    try:
        update = request.get_json(silent=True) or {}

        # Telegram manda muchos tipos de updates, no todos son mensajes
        message = update.get("message") or update.get("edited_message")
        if not message:
            return "ok", 200

        chat = message.get("chat") or {}
        chat_id = chat.get("id")

        text = message.get("text")
        if not chat_id or not text:
            # No es un mensaje de texto → ignorar
            return "ok", 200

        text = text.strip()
        if not text:
            return "ok", 200

        # Comandos básicos
        if text.lower() in ["/start", "/help"]:
            telegram_send_message(
                chat_id,
                "Hola 👋\n\nPuedes preguntarme cosas como:\n"
                "- ¿Qué se dijo sobre el sector energético esta semana?\n"
                "- Compara la primera semana de diciembre vs la segunda\n"
                "- ¿Qué pasó entre el 1 y 7 de diciembre?"
            )
            return "ok", 200

        # Aviso de procesamiento
        telegram_send_message(chat_id, "⏳ Analizando tu pregunta...")

        # 🔒 Protección total: nunca dejar que esto rompa el webhook
        try:
            payload, status = responder_pregunta(text)
        except Exception as e:
            print("❌ Error interno en responder_pregunta:", e)
            telegram_send_message(
                chat_id,
                "⚠️ Ocurrió un error procesando tu pregunta. Intenta reformularla."
            )
            return "ok", 200

        if status != 200 or not isinstance(payload, dict):
            telegram_send_message(
                chat_id,
                "⚠️ No encontré información clara para tu pregunta."
            )
            return "ok", 200

        respuesta = payload.get("respuesta", "")
        if not respuesta:
            telegram_send_message(
                chat_id,
                "⚠️ No encontré información clara para tu pregunta."
            )
            return "ok", 200

        telegram_send_message(chat_id, respuesta[:3500])
        return "ok", 200

    except Exception as e:
        # 🔥 Este catch evita el 500 sí o sí
        print("🔥 Error fatal en /telegram_webhook:", e)
        return "ok", 200


#correoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
@app.route("/enviar_email", methods=["POST"])
def enviar_email():
    data = request.get_json()
    email = data.get("email")
    fecha_str = data.get("fecha")
    fecha_dt = pd.to_datetime(fecha_str).date()

    resultado = generar_resumen_y_datos(fecha_str)
    if "error" in resultado:
        return jsonify({"mensaje": resultado["error"]}), 404

    titulares_info = resultado.get("titulares", [])
    resumen_texto = resultado.get("resumen", "")



    if not resumen_texto:
        archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")
        if os.path.exists(archivo_resumen):
            with open(archivo_resumen, "r", encoding="utf-8") as f:
                resumen_texto = f.read()
        # 🔹 Convertir saltos de línea en HTML para conservar párrafos en el correo
    resumen_html = (resumen_texto or "").replace("\n\n", "<br><br>").replace("\n", "<br>")
    # ☁️ Nube
    archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")

    # ---- CONFIGURACIÓN DEL CORREO ----
    remitente = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_PASS")

    destinatario = email

    msg = MIMEMultipart()
    msg["From"] = formataddr(("Monitoreo +", remitente))  # 👈 nombre visible
    msg["To"] = destinatario
    msg["Subject"] = f"Resumen de noticias {fecha_str}"

    # 📎 Adjuntar logo inline (CID)
    logo_path = os.path.join("static", "Logo_EMAIL.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo = MIMEImage(f.read())
        logo.add_header("Content-ID", "<logo_enfragen>")
        logo.add_header("Content-Disposition", "inline", filename="Logo_EMAIL.png")
        msg.attach(logo)



    # 🧱 Titulares en tabla: máximo 4 por fila (compatible con Gmail/Outlook)
    titulares_cards = []
    for t in titulares_info:
        titulo = (t.get("titulo") or "").strip()
        medio = (t.get("medio") or "").strip()
        enlace = (t.get("enlace") or "").strip()

        card = f"""
        <div style="padding:10px; border:1px solid #ddd; border-radius:12px; background:#fff; height:100%;">
            <a href="{enlace}" style="color:#0B57D0; font-weight:600; text-decoration:none;">
                {titulo}
            </a>
            <br>
            <small style="color:#7D7B78;">• {medio}</small>
        </div>
        """
        titulares_cards.append(card)

    filas_html = []
    for i in range(0, len(titulares_cards), 1):
        fila = titulares_cards[i:i+1]

        # celdas de la fila
        tds = "".join([f'<td style="width:25%; padding:6px; vertical-align:top;">{c}</td>' for c in fila])

        # si faltan celdas para completar 4, rellenar con vacías
        faltan = 1 - len(fila)
        if faltan > 0:
            tds += "".join(['<td style="width:25%; padding:6px;"></td>' for _ in range(faltan)])

        filas_html.append(f"<tr>{tds}</tr>")

    titulares_es_html = f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom:20px; border-collapse:collapse;">
        {''.join(filas_html)}
    </table>
    """

    # 📧 Plantilla HTML con estilo
    cuerpo = f"""
    
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" style="width:100%; max-width:800px; font-family:Montserrat,Arial,sans-serif; border-collapse:collapse; margin:auto;">
    <!-- Header con fondo blanco -->
    <tr>
        <td style="background:#fff; padding:16px 20px; border-bottom:2px solid #e5e7eb;">
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
            <tr>
                <td align="left" style="padding:0; vertical-align:middle;">
                    <img src="cid:logo_enfragen" alt="EnfraGen" style="height:44px; width:auto; display:block;" />
                </td>
                <td align="right" style="font-weight:700; font-size:1.2rem; color:#111; vertical-align:middle;">
                    Monitoreo<span style="color:#FFB429;">+</span>
                </td>
            </tr>
        </table>
        </td>
    </tr>

    <!-- Bloque gris con contenido -->
    <tr>
        <td style="background:#f9f9f9; padding:20px; border:1px solid #e5e7eb; border-radius:0 0 12px 12px;">
        
        <!-- Resumen -->
        <h2 style="font-size:1.4rem; font-weight:700; margin-bottom:14px; color:#111;">
            📅 Resumen diario de noticias — {fecha_str}
        </h2>
        <div style="background:#fff; border:1px solid #ddd; border-radius:12px; padding:20px; margin-bottom:20px;">
            <p style="color:#555; line-height:1.7; text-align:justify;">{resumen_html}</p>
        </div>

        <!-- Titulares español -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">🗞️ Principales titulares</h3>
        {titulares_es_html}
        <!-- Nube -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">☁️ Nube de palabras</h3>
        <div style="text-align:center; margin-top:12px;">
            <img src="cid:nube" alt="Nube de palabras" style="width:100%; max-width:600px; border-radius:12px; border:1px solid #ddd;" />
        </div>

        </td>
    </tr>
    </table>
    """


    msg.attach(MIMEText(cuerpo, "html"))
    
    # 📎 Adjuntar nube inline
    if os.path.exists(archivo_nube):
        with open(archivo_nube, "rb") as img_file:
            imagen = MIMEImage(img_file.read())
            imagen.add_header("Content-ID", "<nube>")
            imagen.add_header("Content-Disposition", "inline", filename=archivo_nube)
            msg.attach(imagen)
      
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Gmail
        server.starttls()
        server.login(remitente, password)
        server.sendmail(remitente, destinatario, msg.as_string())  # 👈 enviar
        server.quit()
        return jsonify({"mensaje": f"✅ Correo enviado a {destinatario}"})
    
    except Exception as e:
    
        return jsonify({"mensaje": f"❌ Error al enviar correo: {e}"})
def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def telegram_send_message_token(bot_token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": False
        # ❌ NO parse_mode
    }
    r = requests.post(url, json=payload, timeout=25)
    r.raise_for_status()
    return r.json()

def telegram_send_photo(bot_token: str, chat_id: str, photo_path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {
            "chat_id": chat_id,
            "caption": (caption or "")[:1024],
            "parse_mode": "HTML",
        }
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.json()

@app.route("/enviar_telegram", methods=["POST"])
def enviar_telegram():
    data = request.get_json() or {}
    fecha_str = (data.get("fecha") or "").strip()
    chat_id = (data.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID_DEFAULT") or "").strip()

    if not fecha_str:
        return jsonify({"mensaje": "❌ Debes enviar 'fecha' (YYYY-MM-DD)."}), 400

    if not chat_id:
        return jsonify({"mensaje": "❌ Debes enviar 'chat_id' o configurar TELEGRAM_CHAT_ID_DEFAULT."}), 400

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return jsonify({"mensaje": "❌ Falta TELEGRAM_BOT_TOKEN en variables de entorno."}), 500

    # 1) Generar resumen y datos (igual que email)
    resultado = generar_resumen_y_datos(fecha_str)
    if "error" in resultado:
        return jsonify({"mensaje": resultado["error"]}), 404

    resumen_texto = (resultado.get("resumen") or "").strip()
    titulares_info = resultado.get("titulares", []) or []

    # 2) Construir mensaje (calca lógica de enviar_email, pero adaptada a Telegram)
    # - Respetar párrafos (como en email)
    # - Mandar TODOS los titulares (no solo 8)
    # - Formatear titulares en filas de 4 (texto/HTML simple de Telegram)

    resumen_texto = (resultado.get("resumen") or "").strip()

    # 🔹 En email conviertes saltos a <br>; en Telegram usamos \n\n para párrafos
    # (escape_html para no romper parse_mode HTML)
    resumen_html = escape_html(resumen_texto).replace("\n\n", "\n\n").replace("\n", "\n")

    # Titulares (TODOS, como email)
    titulares_lines = []
    for t in titulares_info:
        titulo = escape_html((t.get("titulo") or "").strip())
        medio = escape_html((t.get("medio") or "").strip())
        enlace = (t.get("enlace") or "").strip()

        if enlace:
            line = f"• {titulo} ({medio})\n  {enlace}"
        else:
            line = f'• {titulo} <i>({medio})</i>'

        titulares_lines.append(line)

    if titulares_lines:
        titulares_block = "\n".join(titulares_lines)
    else:
        titulares_block = "• (No hay titulares para mostrar)"

    msg = (
        f"📅 Resumen diario — {escape_html(fecha_str)}\n\n"
        f"{resumen_html}\n\n"
        f"🗞️ Principales titulares\n"
        f"{titulares_block}"
    )

    # 3) Enviar texto (Telegram limita ~4096 chars; lo partimos)
    try:
        MAX = 3500
        for i in range(0, len(msg), MAX):
            telegram_send_message_token(bot_token, chat_id, msg[i:i+MAX])
    except Exception as e:
        return jsonify({"mensaje": f"❌ Error enviando mensaje a Telegram: {e}"}), 500

    # 4) Enviar nube como foto (si existe)
    try:
        archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")
        if os.path.exists(archivo_nube):
            telegram_send_photo(
                bot_token,
                chat_id,
                archivo_nube,
                caption=f"☁️ Nube de palabras — {escape_html(fecha_str)}"
            )
        else:
            print(f"⚠️ No existe la nube: {archivo_nube}")
    except Exception as e:
        # No fallamos todo si la foto falla; solo lo reportamos
        print(f"⚠️ Error enviando foto a Telegram: {e}")

    return jsonify({"mensaje": f"✅ Enviado a Telegram (chat_id={chat_id})"})


@app.route("/nube/<filename>")
def serve_nube(filename):
    return send_from_directory("nubes", filename)

@app.route("/fechas", methods=["GET"])
def fechas():
    global df
    try:
        if df.empty:
            print("⚠️ DataFrame vacío al solicitar /fechas")
            return jsonify([])

        # Normalizar tipo de dato (maneja tanto datetime64 como date)
        if pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
            fechas_unicas = df["Fecha"].dropna().dt.date.unique()
        else:
            # Si ya son objetos date o strings convertibles
            fechas_unicas = pd.to_datetime(df["Fecha"], errors="coerce").dropna().dt.date.unique()

        fechas_ordenadas = sorted(fechas_unicas, reverse=True)
        fechas_str = [f.strftime("%Y-%m-%d") for f in fechas_ordenadas]

        print(f"🗓️ /fechas → {len(fechas_str)} fechas detectadas (rango {fechas_str[-1]} → {fechas_str[0]})")
        return jsonify(fechas_str)

    except Exception as e:
        print(f"❌ Error en /fechas: {e}")
        return jsonify([])




# ------------------------------
# 📑 Endpoint para análisis semanal
# ------------------------------
@app.route("/reporte_semanal", methods=["GET"]) 
def reporte_semanal():
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporte_semanal")
    os.makedirs(carpeta, exist_ok=True)

    archivos = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".pdf")
    ]
    archivos.sort(reverse=True)  # más recientes primero

    resultados = []
    for f in archivos:
        # Extraer fechas del nombre (ej: analisis_2025-08-25_a_2025-08-29.pdf)
        match = re.search(r"(\d{4}-\d{2}-\d{2})_a_(\d{4}-\d{2}-\d{2})", f)
        if match:
            fecha_inicio = datetime.strptime(match.group(1), "%Y-%m-%d")
            fecha_fin = datetime.strptime(match.group(2), "%Y-%m-%d")
            nombre_bonito = f"Reporte semanal: {fecha_inicio.day}–{fecha_fin.day} {nombre_mes(fecha_fin)}"
        else:
            nombre_bonito = f  # fallback al nombre del archivo

        resultados.append({
            "nombre": nombre_bonito,
            "url": f"/reporte/{f}"
        })

    return jsonify(resultados)

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/reporte/<path:filename>", methods=["GET"])
def descargar_reporte(filename):
    return send_from_directory("reporte_semanal", filename, as_attachment=False)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
