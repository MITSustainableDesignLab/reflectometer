# reflectometer
This repository contains 3D printable components and software for in-situ measurement of Radiance material reflectance properties. Details are described in Nathaniel L. Jones, Arusha Nirvan, and Christoph Reinhart (2023), [*Low-Cost Photographic Measurement of Colour, Specular Reflectance, and Roughness*](https://www.researchgate.net/publication/373899208_Low-Cost_Photographic_Measurement_of_Colour_Specular_Reflectance_and_Roughness).

## Building the Enclosure
To build the enclosure, manufacture the STL files included in the `stl` directory using a 3D printer. Print `light_enclosure.stl` with white material and the others with black material. The `camera_attachement.stl` piece is sized for a Canon RF 14-35mm f/4 IS USM lens and may be altered to fit other lenses.

Wire LEDs, 3V watch batteries, and a switch into the light enclosure. Affix a light diffuser such as white printer paper to the aperture in the `light_attachement.stl` piece. Cover the insides of the black enclosure pieces with black flock paper to reduce their reflectance.

With the LEDs on, take an HDR photograph of the interior of the box in a room without any other light sources. Use [Photosphere](http://www.anyhere.com/) or similar software to obtain the red, green, and blue radiance values of the diffuser.

## Obtaining HDR images of materials
Place a camera against the `camera_attachement.stl` piece such that it is looking into the enclosure. Tape all seams including around the edge of the camera lens to prevent light leaks.

Place the cross-shaped slot of the enclosure base against a flat piece of the material to be measured. With the LEDs switched on and shinging through the diffuser into the enclosure, take a series of exposures of the sample seen through the enclosure (use autobracketing).

Using [Photosphere](http://www.anyhere.com/) or similar software, convert the image set to an HDR image. Use the same response file that was used to obtain the image of the diffuser. It is recommended to downsample the image to 512 pixels wide.

## Creating the reference images
In the `shell` directory, run `make_reference_image.sh` to generate the reference images. The script takes the following arguments:

| Argument | Meaning |
| --- | --- |
| -p | Generate reference images for Radiance plastic model (default) |
| -a | Generate reference images for Ashikhmin-Shirly model |
| -s | Generate reference image for horizontal slot (default) |
| -t | Generate reference image for vertical slot |
| -f file | Specify the Radiance scene to use (defaults to blackbox.rad) |
| -x resolution | Specify the image resolution (defaults to 512) |
| -vh angle | Specify the horizontal angle of the image in degrees (defaults to 85.431961) |

## Calculating the Radiance parameters
Place the HDR images of all samples to be measured into a directory. In a Python environment, import `reflectometer.py` from the `python` directory and run the following command, which returns output in a DataFrame:

`reflectometer.run(hdr_directory, result_directory, ref_u_path, ref_v_path, diffuser_radiance, sample_count, save_figures, ashik, verbose_errors)`

| Parameter | Type | Meaning |
| --- | --- | --- |
| hdr_directory | string | Path to the directory of input HDR images. |
| result_directory | string, optional | Path to the directory where output should be saved, or None if output should not be saved. If the directory does not exist, it will be created. The default is None. |
| ref_u_path | string, optional | Path to the horizontal reference image, or None if calculations should not be performed. The default is None. |
| ref_v_path | string, optional | Path to the vertical reference image, or None if the material is isotropic. This input is required for the Ashikhmin-Shirley model. The default is None. |
| diffuser_radiance | array-like of numbers, optional | Red, Green, and Blue color channel multipliers, which are the observed radiance of the diffuser in each color channel. The default is [0.118384, 0.109451, 0.121427]. |
| sample_count | integer, optional | If a non-zero value is given, the run will stop after that number of images have been processed. The default is 0. |
| save_figures | boolean, optional | Flag to save images generated during the run to the result_directory, if one is given. The default is False. |
| ashik | boolean, optional | Flag to use Ashikhmin-Shirley model. The default is True. |
| verbose_errors | boolean, optional | Flag to print detailed error messages. The default is False. |
