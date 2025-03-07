# Generador de Arte con GAN

Este proyecto es una aplicación web que utiliza una **Red Generativa Antagónica (GAN)** para generar imágenes de arte de forma automática. Los usuarios pueden ajustar parámetros como el contraste, el tinte y la saturación para personalizar las imágenes generadas. Además, pueden guardar las imágenes generadas en su dispositivo.

## Características principales

- **Generación de imágenes**: Utiliza un modelo GAN ligero para generar imágenes de arte.
- **Ajustes de imagen**: Permite ajustar el contraste, el tinte y la saturación de las imágenes generadas.
- **Guardar imágenes**: Los usuarios pueden guardar las imágenes generadas en formato PNG.
- **Interfaz sencilla**: Una interfaz intuitiva y fácil de usar para generar y personalizar imágenes.

## Tecnologías utilizadas

- **TensorFlow.js**: Para la implementación de la GAN y la generación de imágenes en el navegador.
- **HTML/CSS/JavaScript**: Para la interfaz de usuario y la lógica de la aplicación.
- **Web APIs**: Para la manipulación de imágenes y la interacción con el usuario.

## Cómo usar la aplicación

### Requisitos previos

- Un navegador web moderno (como Chrome, Firefox o Edge).
- Conexión a Internet (para cargar TensorFlow.js).

### Instrucciones

1. **Abrir la aplicación**:
   - Abre el archivo `index.html` en tu navegador.

2. **Generar una imagen**:
   - Haz clic en el botón **"Generar Imagen"** para crear una nueva imagen utilizando el modelo GAN.

3. **Ajustar la imagen**:
   - Utiliza los controles deslizantes para ajustar el **contraste**, el **tinte** y la **saturación** de la imagen generada.
   - Los cambios se aplican en tiempo real.

4. **Guardar la imagen**:
   - Una vez que estés satisfecho con la imagen, haz clic en el botón **"Guardar Imagen"** para descargarla en formato PNG.

### Ejemplo de uso

1. Abre la aplicación en tu navegador.
2. Haz clic en **"Generar Imagen"** para crear una nueva imagen.
3. Ajusta los controles deslizantes para personalizar la imagen.
4. Haz clic en **"Guardar Imagen"** para descargar la imagen generada.

## Estructura del proyecto

- **index.html**: La página principal de la aplicación.
- **script.js**: Contiene la lógica de la GAN, la generación de imágenes y la manipulación de los controles.
- **styles.css**: Estilos CSS para la interfaz de usuario.
- **README.md**: Este archivo, que proporciona información sobre el proyecto.

## Cómo contribuir

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un **fork** del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un **Pull Request** en GitHub.

## Licencia

Este proyecto está bajo la licencia **MIT**. Para más detalles, consulta el archivo [LICENSE](LICENSE).

## Créditos

- **TensorFlow.js**: Para la implementación de la GAN en el navegador.
- **Equipo de desarrollo**: [Tu nombre o equipo].

---

¡Gracias por usar el **Generador de Arte con GAN**! Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue en el repositorio.
