# Forja (nombre de trabajo) · animaciones a medida para cualquier criatura, en local y sin pagar a terceros

Pedido de Daniel, 17/11/2026: una herramienta abierta, ambiciosa, que supere a anymove (patch-gg/anymove) y nos sirva para crear animaciones de **cualquier** criatura que le pongamos delante, sin depender de servicios de pago.

## La idea en una frase
Del modelo 3D de una criatura a un paquete de animaciones listas para el juego: caminar, atacar, recibir golpes, gestos. Sale con sus **contactos y cuadros de impacto anotados** y **físicamente limpias**: pies que no patinan y nada que atraviese. Todo corre en nuestras máquinas.

## Qué toma de anymove y qué le añade

| | anymove | Forja |
|---|---|---|
| Fuente de movimiento | Video IA de pago (MiniMax en fal) | **Cuatro fuentes locales:** video real con teléfonos, video IA local (Wan, Apache-2.0), síntesis procedural y física, y animación a mano asistida por agente |
| Vistas | 2 (frente y lado) | **N vistas** (frente, lado y **arriba**), lo que resuelve qué patas están arriba en criaturas de 8 patas, que es donde anymove falla |
| Caminar con 8 patas | No | **Generador de marcha procedural** a partir de la topología del esqueleto (fases por pata, como nuestra araña) |
| Pies | Patinan 1–2 cm | **Limpieza física con Jolt:** pies clavados, sin atravesar, contacto con el suelo |
| Para el juego | Un GLB | GLB más **anotaciones**: cuadros de contacto, de impacto y de alcance, root motion y ventanas de cancelación. Lo consume qdt-motion (warping e IK al blanco) |
| Otra criatura | Imposible: un lector por modelo | **Retarget por topología** entre criaturas parecidas: araña→escorpión, humano→hormiga humanoide más el par de brazos extra |
| Rig | A mano | **Rig asistido por agente** desde vistas ortográficas, con verificación automática |
| Control de calidad | Puntuación de tomas | **Métricas y crítico:** deslizamiento, penetración, límites articulares, acuerdo entre vistas, lectura visual (critico-de-video) |
| Manejo | Skills de Claude Code | Igual (CLI + skills), para que un agente lo lleve de punta a punta |

## Módulos
1. **`modelo/`:** importar el GLB, escala real, orientación, rig nuevo o existente, colores por extremidad.
2. **`vistas/`:** el diseño de N cámaras en un cuadro, el cuadro de referencia y la calibración. Para video real, un tablero ChArUco impreso o el truco de las paredes ortogonales.
3. **`fuentes/`:**
   - `telefonos`: video real multivista, con MediaPipe/RTMPose para humanos;
   - `video-local`: Wan imagen a video con el cuadro de N vistas;
   - `procedural`: marcha N-patas y gestos por objetivos de IK;
   - `agente`: texto → objetivos de IK y tiempos.
4. **`lector/`:** el lector de articulaciones por criatura (DINOv2, Apache-2.0) entrenado con renders sintéticos de Blender, con batch adaptado a una GPU de 8 GB.
5. **`triangulacion/`:** N vistas a 3D, con mínimos cuadrados, descarte de errores, longitudes de hueso fijas y suavizado temporal.
6. **`ajuste/`:** el rig persigue los puntos, con límites articulares.
7. **`fisica/`:** limpieza con Jolt: contactos, pies, penetración y equilibrio aproximado. Detecta y anota los cuadros de contacto.
8. **`paquete/`:** bucles, root motion, anotaciones y GLB.
9. **`calidad/`:** métricas, crítico y un informe por clip.

## Fases (cada una deja algo que se puede ver)
| Fase | Qué | Prueba de éxito |
|---|---|---|
| F0 | Entorno en el PC viejo (RTX 3070): Python 3.11, torch, Blender 5 portable. Reproducir el ejemplo del dragón de anymove como línea base | GLB del rugido igual al suyo |
| F1 | El lector de **nuestra araña** con batch reducido en 8 GB | Memoria medida y error del examen en cm |
| F2 | Triangulación con **3 vistas**, sobre clips sintéticos renderizados por nosotros (movimiento conocido) | Error 3D < 2 cm y patas arriba/abajo correctas en 8 patas |
| F3 | **Fuente procedural:** marcha N-patas y gestos por IK, más la limpieza física | Araña que camina y se alza sin patinar |
| F4 | **Fuente teléfonos:** dos teléfonos de Daniel con MediaPipe para un gesto humanoide | Un golpe de Daniel aplicado a Proyecto A |
| F5 | **Fuente video local:** Wan en la 3070 con el cuadro de N vistas | Acuerdo entre vistas ≥ 0,9 en un gesto de araña |
| F6 | Anotaciones para el juego y retarget por topología | Clips que qdt-motion consume con su cuadro de impacto |
| F7 | Publicación abierta | Solo con el sí de Daniel, bajo la marca Sendell Games |

## Reglas
- **Licencias:** solo comercial-OK (Apache-2.0, MIT, CC0). Nada de SMPL, AMASS, GVHMR ni modelos con licencias no comerciales. El código tomado de anymove (Apache-2.0) conserva su NOTICE y sus créditos.
- **Sin gastos ni cuentas** sin el sí de Daniel.
- **Lo pesado va al PC viejo,** con OCUPADO.txt; la casita tiene 8 GB.
- **Publicar el repositorio es una decisión de Daniel** (F7). Hasta entonces queda privado y local.

## Riesgos honestos
- **Video IA local en 8 GB:** calidad menor que MiniMax, y puede no mantener las vistas sincronizadas. Por eso no es la única fuente: el procedural y los teléfonos no dependen de él.
- **Entrenar el lector en 8 GB:** probable, pero sin probar (F1 lo responde).
- **Rig de criaturas muy raras:** el agente lo asiste, pero hará falta verificarlo con vistas.
