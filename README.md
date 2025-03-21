# Introduction to Creating RAGs (Retrieval-Augmented Generators) with OpenAI

This lab is designed to introduce students to the fundamental concepts and practical implementation of Retrieval-Augmented Generators (RAGs) using OpenAI’s tools and LangChain framework. By the end of the lab, students will have gained hands-on experience building and understanding RAGs, culminating in the delivery of two GitHub repositories showcasing their work.

## 📌 Características

Este repositorio contiene la implementación de un **Generador Aumentado por Recuperación (RAG)** utilizando las herramientas de OpenAI y el marco LangChain. El proyecto permite integrar la capacidad de recuperación de documentos relevantes a partir de una base de datos de vectores con un modelo generativo de OpenAI, mejorando así las respuestas generadas con contexto relevante.

### Características principales:
- **Integración con Pinecone**: Base de datos de vectores para almacenar y recuperar información relevante.
- **Uso de OpenAI**: Embeddings de OpenAI y LLMs (modelos de lenguaje) para la generación de respuestas.
- **Recuperación de Documentos**: Recuperación eficiente de documentos desde Pinecone en función de una consulta.
- **Generación de Respuestas**: Utilización de un modelo generativo de OpenAI para crear respuestas contextuales basadas en los documentos recuperados.

## 📌 Funcionalidades

- **Carga de documentos**: Carga y procesamiento de documentos desde fuentes web.
- **Generación de embeddings**: Creación de embeddings de los documentos cargados utilizando OpenAI.
- **Búsqueda semántica**: Implementación de un mecanismo de búsqueda semántica para recuperar los documentos más relevantes.
- **Generación de respuestas contextualizadas**: Uso de un modelo de OpenAI para generar respuestas basadas en los documentos recuperados.
- **Interacción con el usuario**: Respuestas personalizadas y contextualizadas en función de las preguntas planteadas.

## 🛠️ Requisitos

Este proyecto está basado en las siguientes tecnologías y librerías. Asegúrate de tener las siguientes dependencias instaladas:

### Librerías principales:
- **LangChain**: Para la construcción del flujo de trabajo del RAG.
- **Pinecone**: Para almacenar y recuperar los vectores generados.
- **OpenAI**: Para generar los embeddings y la respuesta con el modelo de lenguaje.
- **dotenv**: Para gestionar variables de entorno.
- **bs4**: Para el análisis de contenido web (BeautifulSoup).
- **langchain_openai**: Integración con OpenAI a través de LangChain.

### Requisitos de instalación:
1. Python 3.7 o superior.
2. Instalación de las dependencias desde el archivo 


## 🚀 Instalación y Ejecución
### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/Juanse2347/AREP_T8
cd AREP_T8
```


# 🔍 Sistema Arquitectónico
 
Este sistema se basa en una arquitectura distribuida donde se integran varios componentes:

## Capa de Recuperación (Retrieval): ##

Pinecone se utiliza como la base de datos de vectores para almacenar y recuperar documentos relevantes. Los documentos son procesados para generar embeddings usando OpenAI Embeddings.

## Capa Generativa (Generation): ##

OpenAI GPT-4 es utilizado para generar respuestas contextualizadas basadas en los documentos recuperados. La integración se realiza mediante el modelo de OpenAI a través de LangChain.

## Flujo de Datos: ##

- Los datos se cargan desde fuentes web, se convierten en embeddings y se almacenan en Pinecone.
- Cuando se recibe una consulta del usuario, se utiliza el mecanismo de búsqueda de Pinecone para recuperar los documentos más relevantes.
- Estos documentos se pasan al modelo generativo de OpenAI, que genera una respuesta basada en el contenido recuperado.
- Flujo de Trabajo de la Aplicación:

```bash
Carga de datos -> Indexación de embeddings en Pinecone -> Consulta del usuario -> Recuperación de documentos relevantes -> Generación de respuesta.
```



![image](https://github.com/user-attachments/assets/061dd4a4-e740-402a-8a94-665a208ae7bd)




## 🚀 LangChain LLM Chain Tutorial

Instalamos LangSmith

```bash
pip install langchain
```


## Using Language Models

![image](https://github.com/user-attachments/assets/f4b1fa45-7aab-4eb1-b20e-3f7357ded389)


Obtenemos la siguiente respuesta


![image](https://github.com/user-attachments/assets/b71c0ffc-d250-4155-9c57-5303388871c0)


## Prompt Templates


![image](https://github.com/user-attachments/assets/c9cf6938-96ae-4013-9783-2ea478f6c26e)


Obtenemos lo siguiente


![image](https://github.com/user-attachments/assets/31a2a7fd-77bb-49fb-a156-377bb47229a9)


## 🚀 Build a Retrieval Augmented Generation (RAG)

Instalamos Jupyter Notebook

```bash
pip install jupyter notebook
```

## RAG

![image](https://github.com/user-attachments/assets/8aba5a29-3cc6-4b99-b6f2-f65f3400fe20)


![image](https://github.com/user-attachments/assets/e9d8756c-15a3-46de-9e6d-1a426c2475f7)


## LandGraph

![image](https://github.com/user-attachments/assets/70c97acf-e480-48e8-97bb-a2d8cb4230f3)


## Indexing

Codigo

![image](https://github.com/user-attachments/assets/9c8e7df2-daa4-48e0-bee4-fdc4f87dfdc8)


Resultado

![image](https://github.com/user-attachments/assets/4f6efaf0-f23b-43dd-94f0-24e46805ca71)


## Splitting documents

Codigo

![image](https://github.com/user-attachments/assets/3d718454-9e11-4866-bb68-7cfb4209394e)


Resultado

![image](https://github.com/user-attachments/assets/69a25180-2649-4a81-b254-6dc3e0df88c2)


## Storing Documents

Codigo

![image](https://github.com/user-attachments/assets/24bb321a-9077-446e-8b09-4a19aa49b968)


Resultados


![image](https://github.com/user-attachments/assets/2d04748f-1851-4afe-89a7-3e54235f7766)


## Retrieval and Generation

Codigo

![image](https://github.com/user-attachments/assets/8b2c71f5-8747-4a69-acb2-f77a466025c3)


Resultado

![image](https://github.com/user-attachments/assets/8484d27e-b81a-40c1-9c16-dfccf40a1caf)


## LangGraph

Resultado

![image](https://github.com/user-attachments/assets/5e999cc0-8bf4-437b-983d-5e684f399c30)


## Stream Steps

![image](https://github.com/user-attachments/assets/4a4993cb-4928-4282-a232-395f908070bf)








## 🔍 Pruebas de Estilo de Codificacion ##

Con el siguiente comando realizamos las pruebas de estilo de codificación son aquellas que verifican que el código sigue las convenciones y buenas prácticas del equipo o la comunidad

```bash
mvn checkstyle:check
```

![Image](https://github.com/user-attachments/assets/6c5a4c16-9c71-463d-9629-59f5c976213a)


## :office: Desplieqgue ##

Vamos a ejecutar el servidor como un proceso en segundo plano o configurar un servicio systemd, de la siguiente manera:

```bash
mvn spring-boot:run
```

## :cd: Construido con ## 

 - Java - Lenguaje principal utilizado
 - Maven - Para la gestión de dependencias y automatización
 - Docker - Plataforma de código abierto que permite crear, ejecutar, administrar y actualizar contenedores


## :busts_in_silhouette: Contribuciones ##

Lea [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) para obtener detalles sobre nuestro código de conducta y el proceso para enviarnos solicitudes de extracción.

## :school_satchel: Control de Versiones ##

Usamos [SemVer](http://semver.org/) para controlar las versiones.

## :bust_in_silhouette: Autor ##

* **Juan Sebastian Sanchez** - *Trabajo Inicial* - [Juanse2347](https://github.com/Juanse2347)


## 📄 Licencia
Este proyecto está bajo la licencia [LICENSE](LICENSE). ¡Siéntete libre de contribuir! 😊


## :wave: Expresiones de Gratitud ##

- Inspiracion
- Compromiso

