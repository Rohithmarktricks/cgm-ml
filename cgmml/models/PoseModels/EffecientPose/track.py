import helpers
def annotate_image(file_path, coordinates):
    """
    Annotates supplied image from predicted coordinates.
    
    Args:
        file_path: path
            System path of image to annotate
        coordinates: list
            Predicted body part coordinates for image
    """
    
    # Load raw image
    from PIL import Image, ImageDraw
    image = Image.open(file_path)
    image_width, image_height = image.size
    image_side = image_width if image_width >= image_height else image_height

    # Annotate image
    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates[0]
    image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = helpers.display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))
    
    # Save annotated image
    # image.save(normpath(file_path.split('.')[0] + '_tracked.png'))
    return image
