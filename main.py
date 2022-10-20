import DatasetBuilder
import FileIO
import Network_Builder
import Visualize_data

# Set constants here
BATCH_SIZE = 32
image_size = (128, 128, 3)
codings_size = 128
EPOCHS = 10

# Folder location
folder_path = "C:\\Users\\allan\\Downloads\\GANFacesDateset"

# Get all image paths from that folder
image_paths = FileIO.get_list_image_paths(folder_path=folder_path, limit=10000)  # Limit has to be below or equal to 10k

# Convert every single image into an array and return the whole tensor. (Using the image path)
image_tensors = DatasetBuilder.get_image_tensors(image_paths=image_paths, image_size=image_size)

#Visualize_data.display_single_image(image_tensors[10])
#Create a dataset
dataset = DatasetBuilder.build_label_dataset(image_tensors=image_tensors, batch_size=BATCH_SIZE)


network_builder = Network_Builder.Network_Builder(image_size=image_size, codings_size=codings_size)
network_builder.build_generator()
network_builder.build_discriminator()
network_builder.build_GAN()
network_builder.compile_models()
network_builder.summarize_all_models()

network_builder.train_the_network(dataset=dataset, epochs=EPOCHS, codings_size=codings_size)


