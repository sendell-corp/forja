# Crítica de la vitrina F3, primera pasada (17/11/2026)

Video: `%USERPROFILE%\forja\f3\vitrina.mp4` (23,7 s, 960 × 540). Medido con la skill critico-de-video:
`medir.py --fps 4 --telemetria telemetria-segundos.json`. Las 12 hojas, revisadas una por una. Telemetría propia
(`forja/calidad/telemetria.py`): sin `desliza`, sin `atraviesa` ni `bajo-suelo`. Hay `marcha-atras` en tres sitios, todos
pedidos: el peso atrás de alzarse, la carga del mordisco y el retroceso.

| t | qué se ve | regla | gravedad | arreglo |
|---|---|---|---|---|
| 0:08–0:10.75 | «Girar en el sitio»: las patas dan pasos de giro, pero el cuerpo no gira nada en 3 s (debería girar 57°). Parece que patina en el sitio | R3 | **grave** | el importador de glTF deja la armadura en cuaterniones y la vitrina ponía `rotation_euler`: poner `rotation_mode = "XYZ"` (fallo del render, no del clip) |
| 0:18.75–0:21 | «Abalanzarse»: la cámara sigue a la araña y el salto de 0,44 m apenas se nota; el vuelo se lee, el avance no | R8 R11 | media | en los gestos, cámara quieta que encuadre todo el recorrido |
| 0:22.25–0:23.5 | «Retroceder herida»: el ladeo se ve, el retroceso no (la cámara retrocede con ella) | R6 R8 | media | lo mismo: cámara quieta |
| 0:16–0:17.5 | «Mordisco»: desde 3/4 de frente y a esa distancia, el lanzamiento de la cabeza y el cierre de colmillos casi no se distinguen | R8 R2 | media | cámara más de lado y más cerca de la cabeza (en el juego, el blanco lo pone la IK al impacto) |
| todo | las hojas salen estrechas: `medir.py` reduce a 270 × 480 (supone video vertical) y este es 16:9 | — | leve | la criatura se lee igual; proponer en la skill un modo apaisado |

**Resumen:** lo peor, girar no gira (grave, del render); lo mejor, caminar, trotar y alzarse se leen limpios, con pies
clavados, sombra y suelo con textura. No se enseña hasta repetirla con el giro arreglado y la cámara quieta en los gestos.

**Lo que no se pudo juzgar desde las hojas:** si los pies patinan entre dos cuadros (lo cubre la telemetría: 0 cm en
apoyo, ver F3.md) y la costura de los bucles (medida aparte).

**Regla nueva propuesta:** R20 «en una vitrina de animación, la cámara no sigue al cuerpo en los gestos con
desplazamiento: si lo sigue, el desplazamiento no se ve».
