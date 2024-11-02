import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import paho.mqtt.client as mqtt
import json
import time
from PIL import Image
import pytz

# Configuración de la página
st.set_page_config(
    page_title="Sistema Experto CONFORMADORA DE TALONES",
    page_icon="💬",
    layout="wide"
)

# Configuración MQTT
MQTT_BROKER = "broker.mqttdashboard.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Variables de estado para los datos del sensor
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

def get_mqtt_message():
    """Función para obtener un único mensaje MQTT"""
    message_received = {"received": False, "payload": None}
    
    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
            message_received["payload"] = payload
            message_received["received"] = True
        except Exception as e:
            st.error(f"Error al procesar mensaje: {e}")
    
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        
        timeout = time.time() + 5
        while not message_received["received"] and time.time() < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        return message_received["payload"]
    
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# Sidebar
with st.sidebar:
    st.subheader("¿Qué es un sistema Experto?")
    st.write("""
    Este sistema experto te resolverá dudas sobre la conformadora de talones.
    Te ayudará a aprender lo básico sobre la máquina.
    Además, puede interpretar los datos del sensor en tiempo real.
    """)

# Título principal
st.title('Sistema Experto CONFORMADORA DE TALONES💬')

# Cargar imagen
image = Image.open('Instructor.png')
st.image(image)

# Configuración OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"]

# Cargar y procesar PDF
pdfFileObj = open('Temperaturas.pdf', 'rb')
pdf_reader = PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Dividir texto en chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=20,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Crear embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# Columnas para sensor y pregunta
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            st.session_state.sensor_data = sensor_data
            
            if sensor_data:
                st.success("Datos recibidos")
                st.metric("Temperatura", f"{sensor_data.get('Temp', 'N/A')}°C")
                st.metric("Humedad", f"{sensor_data.get('Hum', 'N/A')}%")
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Realiza tu consulta")
    user_question = st.text_area("Escribe tu pregunta aquí:")
    
    if user_question:
        # Incorporar datos del sensor en la pregunta si están disponibles
        if st.session_state.sensor_data:
            enhanced_question = f"""
            Contexto actual del sensor:
            - Temperatura: {st.session_state.sensor_data.get('Temp', 'N/A')}°C
            - Humedad: {st.session_state.sensor_data.get('Hum', 'N/A')}%
            
            Pregunta del usuario:
            {user_question}
            """
        else:
            enhanced_question = user_question
        
        docs = knowledge_base.similarity_search(enhanced_question)
        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with st.spinner('Analizando tu pregunta...'):
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=enhanced_question)
                print(cb)
            
            st.write("Respuesta:", response)

# Cerrar archivo PDF
pdfFileObj.close()
