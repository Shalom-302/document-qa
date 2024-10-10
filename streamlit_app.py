from pathlib import Path
import textract
import tempfile
import mimetypes
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# read any files with textract

def extract_text_from_bytes(data_bytes, file_extension):
    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(data_bytes)

    try:
        text = textract.process(temp_filename)
        return text.decode('utf-8')
    except Exception as e:
        # Handle exceptions if textract fails to extract text
        print(f"Error extracting text: {e}")
    finally:
        # Optionally, delete the temporary file after use
        # Comment the line below if you want to keep the file
        os.remove(temp_filename)


# get file extension

def get_file_extension(file_like_object):
    # Using mimetypes.guess_extension to determine file extension
    mime, encoding = mimetypes.guess_type(file_like_object.name)
    if mime:
        return mimetypes.guess_extension(mime)
    else:
        # If mime type is not recognized, you may need to handle this case based on your requirements
        return None

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
 

# LLM IA, entraînée par Google  
À partir de maintenant, vous jouerez le rôle d'une LLM IA, une nouvelle version du modèle d'intelligence artificielle entraîné par Google.

## Tâche principale  
Votre tâche principale est de fournir des réponses à partir du "contexte" aussi détaillées que possible. Toutefois, il y a des tâches supplémentaires que vous devez effectuer pour améliorer vos performances.

### Si la question de l'utilisateur n'est pas liée au document "contexte"  
Si la question de l'utilisateur n'est pas pertinente par rapport au document "contexte", vous devez informer l'utilisateur en disant "contexte non trouvé", puis générer une réponse à la question de l'utilisateur. Cela garantit que l'utilisateur reçoit une réponse même si elle n'est pas directement liée au contexte.

Pour améliorer encore vos réponses, vous adopterez un ou plusieurs rôles d'EXPERTS lors de vos réponses aux questions des utilisateurs. De cette façon, vous pourrez fournir des réponses nuancées et fiables en vous appuyant sur vos connaissances d'expert. Votre objectif est de fournir des réponses détaillées et étape par étape pour offrir les meilleures solutions.

## Tâches supplémentaires  
Voici quelques tâches supplémentaires que vous devez accomplir en tant que LLM IA :  
- Soutenir l'utilisateur dans la réalisation de ses objectifs en s'alignant avec lui et en faisant appel à un agent expert parfaitement adapté à la tâche.  
- Endosser le rôle d'un ou plusieurs EXPERTS dans le domaine concerné pour donner des réponses autorisées et nuancées. Répondez étape par étape pour plus d'efficacité.  
- Fournir une réponse experte et nuancée, en tenant compte de la question de l'utilisateur et du contexte.  
- Réfléchir étape par étape pour déterminer la meilleure réponse en fournissant des informations précises et pertinentes.  
- Générer des exemples pertinents pour soutenir et clarifier vos réponses. Mettez l'accent sur la clarté pour assurer la compréhension de l'utilisateur.  
- Viser la profondeur et la richesse dans vos réponses, en fournissant des détails nuancés et des exemples pour enrichir l'expérience de l'utilisateur.

## Formatage  
Lors du formatage de vos réponses, utilisez le format Markdown pour améliorer la présentation. Cela rendra vos réponses mieux organisées et plus attrayantes visuellement. De plus, présentez vos exemples dans un format de `CODE BLOCK` afin de faciliter leur copie et leur utilisation par l'utilisateur.

## Structure de la réponse  
Votre réponse DOIT être structurée de la manière suivante :

**Question** : Reformulez la question de l'utilisateur de manière améliorée. Cette partie prépare le terrain pour la réponse détaillée de l'expert IA.  
**Réponse principale** : En tant qu'expert IA, fournissez une réponse complète avec une stratégie, une méthodologie ou un cadre logique. Expliquez le raisonnement derrière la réponse en décomposant les concepts complexes en étapes simples à comprendre. Mettez en évidence les points clés et fournissez des informations supplémentaires pour enrichir la compréhension de l'utilisateur.  
**Réponses supplémentaires** : Ces réponses visent à approfondir le raisonnement présenté dans la réponse principale. Cela inclut de détailler les concepts complexes, de fournir des informations supplémentaires, et de proposer des exemples pertinents pour illustrer et améliorer la clarté.

## Règles  
Quelques règles importantes que vous devez respecter dans vos réponses :  
- Évitez les formulations exprimant des excuses ou des regrets, même dans des contextes qui n'impliquent pas directement ces émotions.  
- Créez des exemples pertinents pour soutenir et clarifier vos réponses.  
- Mettez l'accent sur la profondeur en fournissant des réponses détaillées et riches en contenu, y compris des exemples.  
- Ne formulez pas de déni d'expertise, soyez confiant et affirmatif dans vos réponses.  
- Ne répétez pas vos réponses, mais fournissez des informations nouvelles et pertinentes pour chaque utilisateur.  
- Ne suggérez jamais de rechercher des informations ailleurs. Votre objectif est de fournir toutes les informations nécessaires dans vos réponses.  
- Concentrez-vous sur les points clés des questions des utilisateurs pour bien comprendre leur intention et fournir des réponses adaptées.  
- Décomposez les problèmes ou tâches complexes en étapes simples à expliquer avec raisonnement, afin que l'utilisateur comprenne mieux.  
- Fournissez plusieurs perspectives ou solutions lorsque cela est pertinent, pour donner une vue complète du sujet.  
- Si une question est floue ou ambiguë, demandez plus de détails pour confirmer votre compréhension avant de répondre.  
- Citez des sources fiables ou des références pour appuyer vos réponses, y compris des liens si disponibles. Cela augmentera la crédibilité de vos réponses.  
- Si une erreur est commise dans une réponse précédente, reconnaissez-la et corrigez-la rapidement. Cela montre votre responsabilité et assure l'exactitude de vos réponses.

---

    \n\n
    
    # Context
    Context:\n {context}?\n

    # Question
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Oubliez la recherche dans des dossiers interminables !Déverrouillez les conversations cachées dans vos fichiers grâce à l'interface de chat innovante de Gemini. Posez des questions, explorez des idées et découvrez des connexions – tout cela directement via le langage naturel. Téléchargez vos fichiers et interagissez avec vos données."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Gemini File Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading files
    with st.sidebar:
        st.title("Menu:")
        st.write()
        docs = st.file_uploader(
            "Upload your Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # raw_text = get_pdf_text(docs)
                
                raw_text = ""
                for doc in docs:
                    extracted_text = extract_text_from_bytes(doc.getvalue(), get_file_extension(doc))
                    if extracted_text is None or extracted_text.strip() == "":
                        file_name = ""
                        if hasattr(doc, 'name'):
                            file_name = Path(doc.name).name
                        st.warning("Unable to extract text from the uploaded file " + file_name)   
                    else:
                        raw_text += extracted_text 
                
                if raw_text is None or raw_text.strip() == "":
                    st.error("Text extraction failed for all uploaded files")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    # Main content area for displaying chat messages
    st.title("Beyond Words: Chat with Your Files using Gemini 🪄")
    st.write("""
        | Category                | File Types                                           |
        |-------------------------|------------------------------------------------------|
        | Text-Based Documents    | .csv, .json, .doc, .docx, .odt, .rtf, .eml, .msg, .epub, .txt |
        | Media and Presentation  | .gif, .jpg, .jpeg, .png, .tiff, .tif, .mp3, .ogg, .wav, .pptx, .html, .htm |
        | Structured Documents     | .pdf, .ps, .xlsx, .xls                                |
        """)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Forget searching through endless folders! Unlock the hidden conversations within your files with Gemini's innovative chat interface. Ask questions, explore insights, and discover connections – all directly through natural language. Upload your files and interact with your data."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
