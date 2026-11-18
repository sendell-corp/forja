# Crítica de la vitrina F3, segunda pasada (17/11/2026)

Video: `docs/video/f3_vitrina.mp4` (23,7 s). Mismo método: `medir.py --fps 4` con la telemetría y las 12 hojas, una por una.
Arreglos tras la primera pasada (`critica_vitrina_v1.md`):
- el giro: `rotation_mode = "XYZ"` en la armadura importada;
- cámara quieta en los gestos;
- el mordisco renderizado otra vez con la cámara más de frente y alta. En la primera prueba, con la cámara de lado y cerca, las patas cercanas tapaban la cabeza (R9).

| t | qué se ve | regla | gravedad | arreglo |
|---|---|---|---|---|
| 0:08–0:10.75 | girar: el cuerpo gira (unos 57° en 3 s) y las patas acompañan | — | resuelto | — |
| 0:16–0:17.5 | mordisco: se ven los colmillos abiertos en la carga y cerrados tras lanzar la cabeza, pero el lanzamiento (13 cm) se nota poco a esta escala | R8 | leve | en el juego lo leerá el blanco (IK al impacto); para un video, un plano más cerrado de la cabeza |
| 0:18.75–0:21 | abalanzarse: con la cámara quieta, el salto hacia la cámara se lee | — | resuelto | — |
| 0:22.25–0:23.5 | retroceder herida: se aleja y se ladea; el golpe en sí (cuadro 1) no se ve, porque no hay quien golpee | R2 | leve | es un clip de reacción: en el juego lo dispara el golpe del jugador |
| todo | cartelas de 1 s con `suelo-sin-textura` (son cartelas) y un `cortes-seguidos` en 0:16 (cartela → clip) | R14 | leve | normal en una vitrina |

**Resumen:** sin graves. Lo mejor: marcha, trote y giro con pies clavados, suelo y sombra; alzarse y abalanzarse se leen
claros. Lo flojo: el mordisco y el retroceso son pequeños sin un blanco delante. **Se puede enseñar.**

**Lo que no se pudo juzgar desde las hojas:** el deslizamiento entre cuadros (telemetría: 0) y la costura de los bucles
(medida en F3.md §2).
