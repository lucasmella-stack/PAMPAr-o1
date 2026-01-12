import React, { useRef } from 'react';
import { FileDown } from 'lucide-react';

const PampaRPaperPDF = () => {
  const contentRef = useRef(null);

  const generatePDF = () => {
    const printWindow = window.open('', '_blank');
    const content = contentRef.current.innerHTML;
    
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>PampaR: Una Arquitectura Territorial Inspirada en el Cerebro</title>
        <style>
          @page {
            size: A4;
            margin: 2.5cm;
          }
          
          body {
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #000;
            max-width: 21cm;
            margin: 0 auto;
            padding: 2cm;
          }
          
          h1 {
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5cm;
            line-height: 1.3;
          }
          
          .author {
            text-align: center;
            font-size: 12pt;
            margin-bottom: 0.3cm;
          }
          
          .affiliation {
            text-align: center;
            font-size: 11pt;
            font-style: italic;
            margin-bottom: 0.2cm;
          }
          
          .date {
            text-align: center;
            font-size: 11pt;
            margin-bottom: 1cm;
          }
          
          .abstract {
            background-color: #f8f8f8;
            padding: 0.8cm;
            margin: 1cm 0;
            border-left: 4px solid #333;
          }
          
          .abstract h2 {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 0.5cm;
          }
          
          h2 {
            font-size: 14pt;
            font-weight: bold;
            margin-top: 0.8cm;
            margin-bottom: 0.4cm;
            border-bottom: 1px solid #333;
            padding-bottom: 0.2cm;
          }
          
          h3 {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 0.6cm;
            margin-bottom: 0.3cm;
          }
          
          p {
            text-align: justify;
            margin-bottom: 0.4cm;
          }
          
          .diagram {
            background-color: #f9f9f9;
            padding: 0.6cm;
            margin: 0.6cm 0;
            border: 1px solid #ddd;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            white-space: pre;
            overflow-x: auto;
          }
          
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.6cm 0;
            font-size: 10pt;
          }
          
          th, td {
            border: 1px solid #333;
            padding: 0.3cm;
            text-align: left;
          }
          
          th {
            background-color: #e8e8e8;
            font-weight: bold;
          }
          
          .code {
            background-color: #f5f5f5;
            padding: 0.5cm;
            margin: 0.5cm 0;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            border: 1px solid #ddd;
            overflow-x: auto;
          }
          
          .formula {
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-style: italic;
            margin: 0.5cm 0;
            padding: 0.3cm;
            background-color: #fafafa;
          }
          
          .references {
            font-size: 10pt;
          }
          
          .references ol {
            padding-left: 1.5cm;
          }
          
          .references li {
            margin-bottom: 0.3cm;
          }
          
          ul, ol {
            margin-left: 1cm;
          }
          
          .footer {
            margin-top: 2cm;
            padding-top: 0.5cm;
            border-top: 1px solid #ccc;
            font-size: 9pt;
            text-align: center;
            color: #666;
          }
          
          @media print {
            body {
              padding: 0;
            }
            .no-print {
              display: none;
            }
          }
        </style>
      </head>
      <body>
        ${content}
      </body>
      </html>
    `);
    
    printWindow.document.close();
    setTimeout(() => {
      printWindow.print();
    }, 250);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto bg-white shadow-lg">
        <div className="p-8 no-print flex justify-between items-center bg-blue-600 text-white">
          <div>
            <h1 className="text-2xl font-bold">PampaR Research Paper</h1>
            <p className="text-sm mt-1">Generador de PDF académico</p>
          </div>
          <button
            onClick={generatePDF}
            className="flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
          >
            <FileDown size={20} />
            Descargar PDF
          </button>
        </div>

        <div ref={contentRef} className="p-12 bg-white">
          <h1>
            PampaR: Una Arquitectura Territorial Inspirada en el Cerebro para Modelado de Lenguaje
          </h1>
          
          <div className="author">Lucas Ricardo Mella Chillemi</div>
          <div className="affiliation">Segunda Cabeza</div>
          <div className="date">Enero 2026</div>

          <div className="abstract">
            <h2>Resumen</h2>
            <p>
              Presentamos PampaR, una arquitectura de modelo de lenguaje novedosa inspirada en la organización funcional del cerebro humano. A diferencia de las arquitecturas transformer estándar que tratan todos los cómputos de manera uniforme, PampaR introduce <strong>Procesamiento Territorial</strong> donde módulos neuronales especializados se organizan en territorios funcionales (Expresivo, Contextual, Formal, Estructural) coordinados por un <strong>Tálamo</strong> central que enruta tokens usando un enfoque híbrido: 70% reglas explícitas (LLAVES) y 30% atención aprendida. Nuestros experimentos en WikiText-103 demuestran que esta arquitectura logra perplejidad competitiva (PPL ~57) con solo 14M parámetros, mientras ofrece ventajas de interpretabilidad a través de reglas de enrutamiento explícitas.
            </p>
          </div>

          <h2>1. Introducción</h2>
          <p>
            Los modelos de lenguaje grandes (LLMs) actuales como GPT, LLaMA y Claude logran un rendimiento notable pero sufren de varias limitaciones: (1) <strong>Opacidad</strong>: Los patrones de atención son difíciles de interpretar; (2) <strong>Homogeneidad</strong>: Todas las capas realizan operaciones idénticas; (3) <strong>Ineficiencia</strong>: Todos los tokens son procesados por todos los parámetros.
          </p>
          <p>
            El cerebro humano, en contraste, exhibe clara especialización funcional: el área de Broca para producción del lenguaje, el área de Wernicke para comprensión del lenguaje, la corteza prefrontal para razonamiento lógico, y el hipocampo para memoria y contexto.
          </p>
          <p>
            PampaR (Pampa Reasoning) se inspira en esta organización, implementando una arquitectura territorial donde módulos especializados manejan diferentes aspectos del lenguaje, coordinados por un Tálamo central que enruta información entre módulos usando reglas explícitas (LLAVES) que proporcionan enrutamiento interpretable.
          </p>

          <h2>2. Arquitectura</h2>
          
          <h3>2.1 Vista General</h3>
          <div className="diagram">
     Entrada → Embedding → [Bloque Territorial ×N] → LM Head → Salida
         │                         │
         ▼                         ▼
    ┌────────┐              Tálamo (LLAVES 70% + Atención 30%)
    │        │                     │
    │        │         ┌───────────┴───────────┐
    │        │         │                       │
    │        ▼         ▼                       ▼
    │   ┌─────────────────┐           ┌─────────────────┐
    │   │   EXPRESIVO     │◄─Frontera─┤   CONTEXTUAL    │
    │   │  Lenguaje+Crea  │           │    Contexto     │
    │   └────────┬────────┘           └────────┬────────┘
    │            │                              │
    │            │◄────── Bidireccional ───────►│
    │            │                              │
    │   ┌────────▼────────┐           ┌────────▼────────┐
    │   │     FORMAL      │◄─Frontera─┤  ESTRUCTURAL    │
    │   │     Lógica      │           │  Patrón+Mate    │
    │   └─────────────────┘           └─────────────────┘
    │            │                              │
    └────────────┴──────────────────────────────┘
          </div>

          <h3>2.2 Territorios</h3>
          <p>
            PampaR organiza el cómputo en 4 territorios funcionales, cada uno conteniendo módulos especializados:
          </p>
          <table>
            <thead>
              <tr>
                <th>Territorio</th>
                <th>Módulos</th>
                <th>Función</th>
                <th>Región Cerebral Análoga</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Expresivo</strong></td>
                <td>Lenguaje, Creatividad</td>
                <td>Generación de texto fluido, ideas nuevas</td>
                <td>Área de Broca, Hemisferio derecho</td>
              </tr>
              <tr>
                <td><strong>Contextual</strong></td>
                <td>Contexto</td>
                <td>Memoria de trabajo, coherencia</td>
                <td>Hipocampo, CPF</td>
              </tr>
              <tr>
                <td><strong>Formal</strong></td>
                <td>Lógica</td>
                <td>Razonamiento deductivo, reglas</td>
                <td>Corteza prefrontal</td>
              </tr>
              <tr>
                <td><strong>Estructural</strong></td>
                <td>Patrones, Matemáticas</td>
                <td>Secuencias, números, estructura</td>
                <td>Lóbulo parietal</td>
              </tr>
            </tbody>
          </table>

          <h3>2.3 El Tálamo</h3>
          <p>
            El Tálamo es el orquestador central que enruta tokens a los territorios apropiados. Implementa un mecanismo de enrutamiento híbrido:
          </p>
          <div className="formula">
            pesos_enrutamiento = peso_llaves × activacion_llaves + (1 - peso_llaves) × atencion_aprendida
          </div>
          <p>
            Donde peso_llaves = 0.7 (70% reglas explícitas), activacion_llaves es el enrutamiento basado en reglas del sistema LLAVES, y atencion_aprendida es el enrutamiento aprendido vía mecanismo de atención.
          </p>

          <h3>2.3.1 Sistema de LLAVES</h3>
          <p>
            Las LLAVES son reglas de enrutamiento explícitas e interpretables basadas en patrones de tokens. Esto proporciona interpretabilidad: podemos inspeccionar directamente por qué un token fue enrutado a un territorio específico.
          </p>

          <h3>2.4 Fronteras Bidireccionales</h3>
          <p>
            Los territorios se comunican vía 6 conexiones de frontera bidireccionales con compuertas aprendidas. Cada frontera implementa una transformación donde la salida es el resultado de combinar información de ambos territorios mediante una compuerta aprendida.
          </p>

          <h2>3. Detalles de Implementación</h2>
          
          <h3>3.1 Configuración del Modelo</h3>
          <div className="code">
ConfigPampaR(
    vocab_size=8000,        # Tokenizador BPE
    dim=160,                # Dimensión oculta
    n_heads=4,              # Cabezas de atención por módulo
    n_capas=4,              # Capas por módulo
    dropout=0.1,
    max_seq_len=256,
    peso_llaves=0.7,        # 70% reglas, 30% aprendido
    usar_axiomas=True,      # Habilitar motor de axiomas
    usar_memoria=True,      # Habilitar memoria práctica
)
          </div>
          <p><strong>Parámetros Totales:</strong> 14,069,410 (~14M)</p>

          <h3>3.2 Configuración de Entrenamiento</h3>
          <p>
            Dataset: WikiText-103 (100M tokens). Hardware: NVIDIA GTX 1650 (4GB VRAM). Tamaño de Batch: 4 (efectivo 32 con acumulación de gradientes). Longitud de Secuencia: 128 tokens. Optimizador: AdamW (lr=2e-4, weight_decay=0.01). Precisión: FP16 mixta.
          </p>

          <h2>4. Resultados</h2>
          
          <h3>4.1 Progreso de Entrenamiento</h3>
          <table>
            <thead>
              <tr>
                <th>Fragmento</th>
                <th>Loss Final</th>
                <th>PPL Final</th>
                <th>Mejora</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1 (10M)</td>
                <td>4.85</td>
                <td>127.5</td>
                <td>Línea base</td>
              </tr>
              <tr>
                <td>2 (20M)</td>
                <td>4.22</td>
                <td>68.1</td>
                <td>-46.6% PPL</td>
              </tr>
              <tr>
                <td>3 (35M)</td>
                <td>~4.05</td>
                <td>~57.1</td>
                <td>-55.2% PPL</td>
              </tr>
            </tbody>
          </table>

          <h3>4.2 Comparación con Líneas Base</h3>
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Parámetros</th>
                <th>PPL (WikiText-103)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>LSTM (Merity et al.)</td>
                <td>24M</td>
                <td>69.1</td>
              </tr>
              <tr>
                <td>Transformer-XL</td>
                <td>24M</td>
                <td>54.5</td>
              </tr>
              <tr>
                <td><strong>PampaR v9</strong></td>
                <td><strong>14M</strong></td>
                <td><strong>~57</strong></td>
              </tr>
              <tr>
                <td>GPT-2 Small</td>
                <td>125M</td>
                <td>35.1</td>
              </tr>
            </tbody>
          </table>
          <p>
            PampaR logra perplejidad competitiva con 40% menos parámetros que modelos LSTM comparables.
          </p>

          <h2>5. Discusión</h2>
          
          <h3>5.1 Ventajas</h3>
          <p>
            (1) <strong>Interpretabilidad</strong>: LLAVES proporcionan fundamento explícito de enrutamiento. (2) <strong>Eficiencia de Parámetros</strong>: PPL competitivo con menos parámetros. (3) <strong>Modularidad</strong>: Fácil agregar/modificar territorios especializados. (4) <strong>Plausibilidad Biológica</strong>: Refleja organización funcional del cerebro.
          </p>

          <h3>5.2 Limitaciones</h3>
          <p>
            (1) <strong>Escala</strong>: Aún no probado a escala de 1B+ parámetros. (2) <strong>Tareas</strong>: Evaluado solo en modelado de lenguaje (PPL). (3) <strong>Diseño de LLAVES</strong>: Actualmente manual, podría ser aprendido.
          </p>

          <h3>5.3 Trabajo Futuro</h3>
          <p>
            Escalamiento a 1B, 7B parámetros; evaluación en razonamiento, QA, generación de código; aprendizaje automático de LLAVES; y comparación de activaciones con datos de neuroimagen.
          </p>

          <h2>6. Conclusión</h2>
          <p>
            PampaR demuestra que arquitecturas territoriales inspiradas en el cerebro pueden lograr rendimiento competitivo en modelado de lenguaje mientras proporcionan ventajas de interpretabilidad. La combinación de reglas explícitas (LLAVES) con atención aprendida ofrece una dirección prometedora para construir sistemas de IA más transparentes. La arquitectura es código abierto bajo licencia AGPL-3.0, habilitando colaboración comunitaria e investigación adicional.
          </p>

          <div className="references">
            <h2>Referencias</h2>
            <ol>
              <li>Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.</li>
              <li>Merity, S., et al. (2018). Regularizing and Optimizing LSTM Language Models.</li>
              <li>Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.</li>
              <li>Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla).</li>
              <li>Fedorenko, E., & Thompson-Schill, S. L. (2014). Reworking the language network. Trends in Cognitive Sciences.</li>
            </ol>
          </div>

          <h2>Apéndice A: Reproducibilidad</h2>
          <p>
            <strong>Repositorio de Código:</strong><br/>
            GitHub: https://github.com/lucasmella-stack/llarri-o1<br/>
            HuggingFace: https://huggingface.co/lucas-mella/PAMPAr-o1
          </p>

          <div className="footer">
            <strong>Licencia:</strong> AGPL-3.0-or-later<br/>
            <strong>Copyright:</strong> © 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
          </div>
        </div>
      </div>
    </div>
  );
};

export default PampaRPaperPDF;