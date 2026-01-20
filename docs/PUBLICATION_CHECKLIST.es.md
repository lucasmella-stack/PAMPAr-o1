# 📋 Checklist de Publicación - PampaR v9

Este documento te guía paso a paso para publicar PampaR de forma profesional.

---

## ✅ Fase 1: Preparación del Repositorio (YA COMPLETADO)

- [x] Licencia AGPL-3.0-or-later (archivo `LICENSE`)
- [x] README.md completo con badges, arquitectura, resultados
- [x] README.es.md versión en español
- [x] CITATION.cff para citas académicas
- [x] CONTRIBUTING.md guía de contribución
- [x] SECURITY.md política de seguridad
- [x] CODEOWNERS para gestión de PRs
- [x] .gitignore configurado correctamente
- [x] pyproject.toml con metadatos del paquete
- [x] requirements.txt con dependencias
- [x] Documentación técnica en `docs/`
- [x] Diagramas de arquitectura en `diagrams/`
- [x] Scripts de entrenamiento e inferencia
- [x] Tests básicos en `tests/`
- [x] Ejemplos de uso en `examples/`

---

## 🔄 Fase 2: Limpieza Pre-Publicación

### 2.1 Verificar que NO se suban archivos sensibles
```bash
# Verificar qué archivos están trackeados
git ls-files

# Asegurarte de que checkpoints NO están incluidos
git ls-files | grep -E "\.(pt|bin|safetensors)$"
# (Debería dar vacío)
```

### 2.2 Verificar .gitignore
- [x] `checkpoints/*.pt` excluido
- [x] `data/wikitext-103/` excluido
- [x] `data/MNIST/` excluido
- [x] `*.log` excluido
- [x] Secretos excluidos (`*.pem`, `*.key`, `secrets/`)

### 2.3 Limpiar historial si es necesario
```bash
# Si subiste archivos grandes por error:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch checkpoints/*.pt" \
  --prune-empty --tag-name-filter cat -- --all
```

---

## 🌐 Fase 3: Publicación en GitHub

### 3.1 Hacer el repositorio público
1. Ve a **Settings** → **General** → **Danger Zone**
2. Click en **Change visibility**
3. Selecciona **Public**
4. Confirma escribiendo el nombre del repo

### 3.2 Configurar GitHub Pages (opcional, para docs)
1. Ve a **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / `docs`
4. Tu documentación estará en `https://lucasmella-stack.github.io/PAMPAr-o1/`

### 3.3 Crear Release en GitHub
1. Ve a **Releases** → **Create a new release**
2. Tag: `v9.0.0`
3. Title: `PampaR v9.0.0 - Arquitectura Territorial`
4. Descripción:
```markdown
## 🦙 PampaR v9.0.0 - Arquitectura Territorial

### Destacados
• 14M parámetros, PPL ~45 en WikiText-103
• Nueva arquitectura territorial con 4 territorios especializados
• Routing híbrido: 70% LLAVES (reglas explícitas) + 30% atención aprendida
• Entrenado en hardware consumer (GTX 1650 4GB)

### Qué incluye
- Código fuente completo bajo AGPL-3.0
- Scripts de entrenamiento e inferencia
- Documentación y paper técnico
- Diagramas de arquitectura

### Nota
Los checkpoints NO están incluidos por tamaño. Entrenar con:
```bash
python scripts/train.py --tokens 50M --epochs 10
```

### Cita
Ver CITATION.cff para cómo citar este trabajo.
```
5. Click **Publish release**

---

## 📚 Fase 4: Obtener DOI con Zenodo

### 4.1 Conectar GitHub con Zenodo
1. Ve a https://zenodo.org
2. Click **Log in** → **Log in with GitHub**
3. Autoriza Zenodo
4. Ve a **GitHub** en la barra lateral
5. Busca `lucasmella-stack/PAMPAr-o1`
6. Activa el toggle ✅

### 4.2 Crear Release en GitHub (triggerea Zenodo)
- Cuando crees un release en GitHub, Zenodo automáticamente:
  - Archiva el código
  - Genera un DOI
  - Lee `.zenodo.json` para metadatos

### 4.3 Obtener tu DOI
1. Ve a https://zenodo.org/account/settings/github/
2. Busca tu repositorio
3. Copia el DOI badge
4. Actualiza README.md con el badge

---

## 📝 Fase 5: Publicar Preprint en arXiv

### 5.1 Crear cuenta en arXiv
1. Ve a https://arxiv.org/user/register
2. Completa el registro
3. Espera verificación (puede tardar 1-2 días)

### 5.2 Preparar el paper
- Ya tienes `docs/PAPER_TECNICO.md`
- Puedes convertirlo a LaTeX o subir PDF directamente
- El abstract está en `docs/ARXIV_ABSTRACT.md`

### 5.3 Enviar a arXiv
1. Ve a https://arxiv.org/submit
2. Categoría primaria: **cs.CL** (Computation and Language)
3. Cross-list: **cs.AI**, **cs.LG**
4. Sube archivos
5. Completa metadatos
6. Envía para moderación

### 5.4 Después de aceptación
- Recibirás ID como `arXiv:2601.XXXXX`
- Actualiza CITATION.cff
- Actualiza README con badge de arXiv

---

## 💼 Fase 6: Visibilidad Profesional

### 6.1 Post de LinkedIn (ejemplo)
```
🚀 Proyecto Open Source: PampaR - Modelo de Lenguaje con Arquitectura Cerebral

Después de meses de desarrollo, publico PampaR, un modelo de lenguaje con arquitectura inspirada en el cerebro humano.

📊 Resultados:
• 14M parámetros (42% menos que modelos comparables)
• Perplexity ~45 en WikiText-103
• Supera LSTM y Transformer-XL Small
• Entrenado en hardware consumer (GTX 1650 4GB)

🧠 Innovaciones:
• 4 territorios especializados (Expresivo, Contextual, Formal, Estructural)
• Sistema LLAVES: 70% reglas explícitas + 30% atención aprendida
• Interpretabilidad: podemos ver qué territorio procesa cada token

🔗 Código y documentación: [GitHub Link]
📄 Paper técnico: [arXiv Link]

Desarrollado como proyecto independiente, demostrando que la innovación en IA no requiere millones en GPUs.

#MachineLearning #AI #OpenSource #NLP #Research

¿Preguntas? ¡Feliz de discutir la arquitectura!
```

### 6.2 Actualizar perfil de GitHub
- Agrega PampaR como proyecto destacado (pin)
- Actualiza bio mencionando IA/ML

### 6.3 Contactar universidades (template)
```
Asunto: Propuesta de colaboración - Arquitectura Territorial para Modelos de Lenguaje

Estimado/a [Nombre del Profesor],

Me llamo Lucas Mella Chillemi y he desarrollado PampaR, una arquitectura de modelo de lenguaje inspirada en la organización funcional del cerebro humano.

El modelo logra perplexity ~45 con solo 14M de parámetros en WikiText-103, superando baselines como LSTM (24M, PPL 69.1) y Transformer-XL Small (24M, PPL 54.5). Todo el entrenamiento se realizó en hardware consumer (GTX 1650 4GB).

La innovación principal es el sistema de "territorios" con routing híbrido (reglas explícitas + atención aprendida), que ofrece ventajas en interpretabilidad y eficiencia.

El código está disponible públicamente bajo AGPL-3.0:
- GitHub: [link]
- Paper: [arXiv link]

Me interesaría discutir posibles colaboraciones de investigación o la posibilidad de presentar este trabajo en su grupo.

Quedo atento a su respuesta.

Saludos cordiales,
Lucas Ricardo Mella Chillemi
```

---

## 🎯 Resumen de Próximos Pasos

| Paso | Prioridad | Tiempo Estimado |
|------|-----------|-----------------|
| 1. Hacer repo público | 🔴 Alta | 5 min |
| 2. Crear Release v9.0.0 | 🔴 Alta | 15 min |
| 3. Conectar Zenodo → DOI | 🔴 Alta | 10 min |
| 4. Crear cuenta arXiv | 🟡 Media | 10 min |
| 5. Enviar preprint | 🟡 Media | 30 min |
| 6. Post LinkedIn | 🟡 Media | 20 min |
| 7. Contactar universidades | 🟢 Baja | Variable |

---

## 📎 Archivos Relacionados

- `.zenodo.json` - Metadatos para Zenodo DOI
- `docs/ARXIV_ABSTRACT.md` - Abstract y checklist para arXiv
- `CITATION.cff` - Información para citas académicas
- `docs/PAPER_TECNICO.md` - Paper técnico completo
- `docs/PAPER_TECNICO.es.md` - Paper técnico en español

---

*Última actualización: Enero 2026*
