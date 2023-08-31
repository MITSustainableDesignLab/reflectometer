# -*- coding: utf-8 -*-
"""
This code performs measurement of Radiance material reflectance properties
Details are described in Nathaniel L. Jones, Arusha Nirvan, and Christoph
Reinhart (2023), Low-Cost Photographic Measurement of Colour, Specular
Reflectance, and Roughness, published at Building Simulation 2023.

Use the run() command at the end of this file to run the script.

@author: Nathaniel Jones
"""

import os
import traceback
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

WHITE_EFFICACY = 179 # lumens per watt

mpl.rc("figure", figsize=(6,4), dpi=150)

def read(path, out=None, dtype=np.float32, ward=False, progress=False, flatten_header=False):
    """
    Read a Radiance image file.
    Adapted from https://www.graphics.cornell.edu/~bjw/rgbe/rgbe.c

    Parameters
    ----------
    path : string
        Path to the Radiance image file.
    out : ndarray, optional
        Target array to write image to, which must have the same dimensions as
        the image, or None to create a new array. The default is None.
    dtype : type, optional
        Data type of the created image. The default is 32-bit float. This is
        ignored if a target is provided in 'out'.
    ward : boolean, optional
        Flag to use Ward's rounding scheme if True, or to preserve mapping of
        [0.1] range if False. The default is False.
    flatten_header : boolean, optional
        Flag to remove leading whitespace from header entries. The default is False.

    Raises
    ------
    ValueError
        A problem reading the file.

    Returns
    -------
    image : ndarray
        Matrix containing the image. The outer dimensions are pixel rows and
        columns, and the innermost dimension holds red, green, and blue channels.
    header : dict
        Header information from file.

    """
    with open(path, 'rb') as reader:
        # Read the file header
        header = {}
        exposure = 1
        EXPOSURE = 'EXPOSURE'
        for line in reader:
            decoded = line.decode('ascii')
            if decoded.isspace(): # end of header
                break

            entries = decoded.split('=', 1)
            if len(entries) == 2:
                if flatten_header:
                    header.update({entries[0].lstrip(): entries[1].strip()})
                else:
                    header.update({entries[0]: entries[1].strip()})
                if entries[0] == EXPOSURE:
                    exposure * float(header[EXPOSURE])

        # Check format
        formt = header['FORMAT']
        if formt is None or formt.lower() != '32-bit_rle_rgbe':
            raise ValueError("Unrecognized format: %s" % formt)

        # Check header values
        if exposure <= 0:
            raise ValueError("Exposure must be positive: %g" % exposure)

        # Read resolution string
        decoded = reader.readline().decode('ascii').strip()
        entries = decoded.split()
        if len(entries) != 4:
            raise ValueError('File %s has malformed resolution %s' % (path, decoded))
        rows = int(entries[1])
        cols = int(entries[3])

        # Check output array
        if out is not None and out.shape == (rows, cols, 3):
            out.fill(0)
        else:
            out = np.zeros((rows, cols, 3), dtype=dtype)

        # Read image data
        rgbe = np.fromfile(reader, dtype=np.ubyte)
        ptr = 0
        for row in range(rows):
            if rgbe[ptr] != 2 or rgbe[ptr + 1] != 2 or rgbe[ptr + 2] & 0x80:
                # not row length encoded
                scanline_buffer = rgbe[ptr:ptr + cols * 4].reshape((cols, 4))
                ptr += cols * 4
            else:
                # row length encoded
                scanline_width = rgbe[ptr + 2] << 8 | rgbe[ptr + 3]
                scanline_buffer = np.empty(scanline_width * 4, dtype=np.ubyte)
                si = 0
                ptr += 4
    
                while si < scanline_buffer.size:
                    if rgbe[ptr] > 128:
                        # a run of the same value
                        count = rgbe[ptr] - 128
                        scanline_buffer[si:si + count] = rgbe[ptr + 1]
                        si += count
                        ptr += 2
                    else:
                        # a non-run
                        count = rgbe[ptr]
                        scanline_buffer[si:si + count] = rgbe[ptr + 1:ptr + 1 + count]
                        si += count
                        ptr += 1 + count
                scanline_buffer = scanline_buffer.reshape((4, scanline_width)).T

            # convert data from buffer into floats
            nonz = scanline_buffer[:,3] != 0
            if nonz.any():
                f = np.ldexp(1.0, scanline_buffer[nonz, 3].astype(np.int32) - (128 + 8)).astype(out.dtype)
                if ward:
                    out[row, nonz, :] = (scanline_buffer[nonz, :3] + np.float32(0.5)) * f[..., np.newaxis]
                else:
                    out[row, nonz, :] = scanline_buffer[nonz, :3] * f[..., np.newaxis]

    # Correct exposure
    if exposure != 1:
        out /= exposure

    return out, header

def bright(rgb, camera=False):
    """
    Calculate the combined brightness of Radiance color chanels

    Parameters
    ----------
    rgb : array_like
        Matrix with color chanels in last dimension.
    camera : boolean, optional
        Flag indicating whether CCIR-709 primaries are provided in the
        chanels. The default is False.

    Returns
    -------
    array_like
        The combined brighness for each set of input RGB values.

    """
    if camera:
        return np.dot(rgb, [0.213, 0.715, 0.072])
    return np.dot(rgb, [0.265, 0.67, 0.065])

def to_ldr(hdr, gamma=2.2, black=0, white=None):
    """
    Convert a high dynamic range image to the color space [0 - 1].

    Parameters
    ----------
    hdr : ndarray
        High dynamic range image.
    gamma : number, optional
        The gamma correction factor. The default is 2.2.
    black : number, optional
        The maximum brightness value in the original image that will be
        rendered as black in the output. The default is 0.
    white : number, optional
        The minimum brightness value in the original image that will be
        rendered as white in the output, or None to use the maximum brightness
        found in the image. The default is None.

    Returns
    -------
    ldr : ndarray
        The image mapped to the [0 - 1] color space.

    """
    if white is None:
        white = hdr.max()
        if black != 0:
            hdr = np.maximum(hdr - black, 0)
    else:
        white -= black
        hdr = np.clip(hdr - black, 0, white)
    if white > 0:
        ldr = (hdr/white) ** (1/gamma)
    else:
        ldr = np.zeros_like(hdr)
    return ldr

def slotY(hdr, x, window=0.25, min_brightness=0.001):
    """
    Find the center pixel y-coordinate of the slot.

    Parameters
    ----------
    hdr : ndarray
        High dynamic range image.
    x : int
        The x-coordinate of the image to scan.
    window : float
        The fraction of the vertical scanline to consider.
    min_brightness : number, optional
        The minimum pixel brightness that could be in part of the sample. The
        default is 0.001.

    Returns
    -------
    percentile50 : float
        The likely center of the slot in the image.

    """
    # Select search window
    window = np.clip(window, 0, 1)
    y_min = int(hdr.shape[0] / 2 * (1 - window))
    y_max = hdr.shape[0] - y_min
    vertical_scanline = bright(hdr[y_min:y_max, x], camera=True) * WHITE_EFFICACY
    
    # Find pixel representing 50th percentile of luminance cumulative density function
    cdf = np.cumsum(vertical_scanline) / vertical_scanline.sum()
    percentile50 = int(np.interp(0.5, cdf, np.arange(cdf.size)))
    #return percentile50 + y_min
    
    # Find edges
    sample_top_y = np.argwhere(vertical_scanline[:percentile50] < min_brightness)
    sample_bottom_y = np.argwhere(vertical_scanline[percentile50:] < min_brightness)
    if sample_top_y.size == 0 or sample_bottom_y.size == 0:
        return percentile50 + y_min
    sample_top_y = sample_top_y[-1, 0] + 1 + y_min
    sample_bottom_y = sample_bottom_y[0, 0] - 1 + percentile50 + y_min

    return (sample_top_y + sample_bottom_y) / 2

def slotFromScanline(scanline, slot_width, pixels_per_mm):
    """
    Identify the slot based on the maximum brightness region in the scanline.

    Parameters
    ----------
    scanline : ndarray
        Pixel brightnesses in the scanline.
    slot_width : number
        Width of the slot in millimeters.
    pixels_per_mm : number
        Image scale in pixels per millimeter.

    Returns
    -------
    xmin : integer
        Pixel index of left edge of slot.
    xmax : integer
        Pixel index of right edge of slot.

    """
    slot_pixels = round(slot_width * pixels_per_mm)
    cumulative_sum = np.cumsum(scanline)
    window_sums = cumulative_sum[slot_pixels:] - cumulative_sum[:-slot_pixels]
    xmin = np.flatnonzero(window_sums == window_sums.max())
    if not np.isscalar(xmin):
        xmin = round(xmin.mean())
    return xmin, xmin + slot_pixels

def reference_image(ref_photo_path, diffuser_radiance, ashik=True, transverse=False, destination=None, name='Reference'):
    """
    Read the reference image.

    Parameters
    ----------
    ref_photo_path : string
        Path to reference image.
    diffuser_radiance : array-like
        Red, Green, and Blue color channel multipliers, which are the observed
        radiance of the diffuser in each color channel.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley reference. The default is True.
    transverse : boolean, optional
        Flag to treat image as transverse reference. The default is False.
    destination : string, optional
        Directory path for saving plots, or None if no plots shoud not be
        saved. The default is None.
    name : string, optional
        Name of reference image for display. The default is 'Reference'.

    Returns
    -------
    luminance : ndarray
        HDR image of reference scaled to luminance units.

    """
    hdr, header = read(ref_photo_path, flatten_header=True)
    view_args = header['VIEW'].split()
    horizontal_angle = float(view_args[view_args.index('-vh') + 1])

    camera = Camera()
    photo_half_width = camera.focal_height * np.tan(np.deg2rad(horizontal_angle / 2))
    pixels_per_mm = hdr.shape[1] / (2 * photo_half_width)

    # Get sample extents
    if transverse:
        sample_left_pix = round((hdr.shape[1] - camera.crosshair_length * pixels_per_mm) / 2)
        sample_right_pix = round((hdr.shape[1] + camera.crosshair_length * pixels_per_mm) / 2)
        markers = [ (sample_left_pix + sample_right_pix) / (2 * pixels_per_mm) ]
    else:
        sample_left_pix = round((photo_half_width - camera.slot_left) * pixels_per_mm)
        sample_right_pix = round((photo_half_width + camera.slot_right) * pixels_per_mm)
        markers = [ photo_half_width, photo_half_width - camera.slot_light, photo_half_width - camera.slot_highlight ]
    
    luminance = hdr * WHITE_EFFICACY * np.array(diffuser_radiance)
    n_rows = luminance.shape[0] - 1

    # Show luminance section
    x = np.arange(sample_left_pix, sample_right_pix) / pixels_per_mm
    plt.plot(x, luminance[-1, sample_left_pix:sample_right_pix, 0], label = 'Diffuse')
    if ashik:
        sample_rows = [0, 125, 250]
    else:
        sample_rows = [300, 200, 100, 0]
    for row in sample_rows:
        plt.plot(x, luminance[row, sample_left_pix:sample_right_pix, 0], label = 'Rough %g' % roughnessLookUp(row, n_rows, ashik=ashik))
    for marker in markers:
        plt.axvline(x = marker, linestyle=':')

    # Plot decorations
    plotter = Plotter(name, camera.slot_left + camera.slot_right, destination=destination)
    plotter.plot()

    return luminance

def rotateHDR(hdr, pixel_inset=10, window=0.25, min_rotation=0.2, min_brightness=0.001, plot=False):
    """
    Rotate the HDR image to horizontal

    Parameters
    ----------
    hdr : ndarray
        Input image.
    pixel_inset : number, optional
        Distance to offset from detected edge of sample. The default is 10.
    window : number, optional
        The fraction of the vertical scanline to consider. The default is 0.25.
    min_rotation : number, optional
        The minimum rotation angle to prompt a rotation operation to be
        performed on the image. The default is 0.2.
    min_brightness : number, optional
        The minimum pixel brightness that could be in part of the sample. The
        default is 0.001.
    plot : boolean, optional
        Flag to plot the image for debugging. The default is False.

    Returns
    -------
    hdr : ndarray
        Rotated output image.

    """
    # Define search window
    window = np.clip(window, 0, 1)
    y_min = int(hdr.shape[0] / 2 * (1 - window))
    y_max = hdr.shape[0] - y_min
    mean_lum = bright(hdr[y_min:y_max], camera=True).mean(axis=0) * WHITE_EFFICACY

    # Find sample extents
    max_luminance_x = np.argmax(mean_lum)
    sample_left_x = np.argwhere(mean_lum[:max_luminance_x] < min_brightness)
    sample_right_x = np.argwhere(mean_lum[max_luminance_x:] < min_brightness)
    if sample_left_x.size == 0:
        sample_left_x = pixel_inset
    else:
        sample_left_x = sample_left_x[-1, 0] + pixel_inset
    if sample_right_x.size == 0:
        sample_right_x = mean_lum.size - pixel_inset
    else:
        sample_right_x = sample_right_x[0, 0] + max_luminance_x - pixel_inset
    sample_left_y = slotY(hdr, sample_left_x)
    sample_right_y = slotY(hdr, sample_right_x)

    if plot:
        # Show photograph
        plt.imshow(to_ldr(hdr, gamma=3.5, white=1))
        plt.plot([max_luminance_x, max_luminance_x], [0, hdr.shape[0] - 1])
        plt.plot(sample_left_x, sample_left_y, marker="o", markersize=3)
        plt.plot(sample_right_x, sample_right_y, marker="o", markersize=3)
        plt.show()

    # Rotate if necessary
    rotation = np.degrees(np.arctan2(sample_right_y - sample_left_y, sample_right_x - sample_left_x))
    if rotation > min_rotation or rotation < -min_rotation:
        hdr = ndimage.rotate(hdr, rotation, reshape=False)
    return hdr

class CameraAlignment:
    def __init__(self, camera, pixels_per_mm):
        """
        This is a class for parameters that describe the location of the sample
        within an HDR image.

        Parameters
        ----------
        camera : Camera
            Geometric desicription of the camera.
        pixels_per_mm : number
            Image scale in pixels per millimeter.

        Returns
        -------
        None.

        """
        self.camera = camera
        self.pixels_per_mm = pixels_per_mm
        self.x_offset = 0

    def findSlotY(self, hdr, window=0.125, min_brightness=0.001, pixel_inset=10, plot=False):
        """
        Find the vertical location of the horizontal slot.

        Parameters
        ----------
        hdr : ndarray
            HDR image.
        window : number, optional
            The fraction of the vertical scanline to consider. The default is 0.125.
        min_brightness : number, optional
            The minimum pixel brightness that could be in part of the sample. The
            default is 0.001.
        pixel_inset : number, optional
            Distance to offset from detected edge of sample. The default is 10.
        plot : boolean, optional
            Flag to plot the scanline for debugging. The default is False.

        Returns
        -------
        None.

        """
        # Define search window
        window = np.clip(window, 0, 1)
        y_min = int(hdr.shape[0] / 2 * (1 - window))
        y_max = hdr.shape[0] - y_min
        mean_lum = bright(hdr[y_min:y_max], camera=True).mean(axis=0) * WHITE_EFFICACY

        # Find sample extents
        max_luminance_x = np.argmax(mean_lum)
        sample_left_x = np.argwhere(mean_lum[:max_luminance_x] < min_brightness)
        sample_right_x = np.argwhere(mean_lum[max_luminance_x:] < min_brightness)
        if sample_left_x.size == 0:
            sample_left_x = pixel_inset
        else:
            sample_left_x = sample_left_x[-1, 0] + pixel_inset
        if sample_right_x.size == 0:
            sample_right_x = mean_lum.size - pixel_inset
        else:
            sample_right_x = sample_right_x[0, 0] + max_luminance_x - pixel_inset

        # Find point close to center
        under_camera_x = hdr.shape[1] // 2
        under_camera_x = np.clip(under_camera_x, sample_left_x, sample_right_x)
        slot_center_y = slotY(hdr, under_camera_x)

        # Find slot extents in photograph
        self.slot_photo_y_min = round(slot_center_y - self.camera.slot_width * self.pixels_per_mm / 2)
        self.slot_photo_y_max = round(slot_center_y + self.camera.slot_width * self.pixels_per_mm / 2)

        if plot:
            # Show photograph
            plt.imshow(to_ldr(hdr, gamma=3.5, white=1))
            plt.plot([sample_left_x, sample_left_x], [0, hdr.shape[0] - 1])
            plt.plot([sample_right_x, sample_right_x], [0, hdr.shape[0] - 1])
            plt.plot(under_camera_x, slot_center_y, marker="o", markersize=3)
            plt.show()

    def slotRangeY(self):
        """ The vertical pixel range containing the horizontal slot. """
        return slice(self.slot_photo_y_min, self.slot_photo_y_max)

    def findCrosshairX(self, hdr, pixel_inset=5, crosshair_search_min=100, crosshair_search_max=200, plot=False):
        """
        Find the horizontal location of the vertical slot.

        Parameters
        ----------
        hdr : ndarray
            HDR image.
        pixel_inset : integer, optional
            Distance to offset from detected edge of sample. The default is 5.
        crosshair_search_min : integer, optional
            Minimum horizontal pixel value for search. The default is 100.
        crosshair_search_max : integer, optional
            Maximum horizontal pixel value for search. The default is 200.
        plot : boolean, optional
            Flag to plot the scanline for debugging. The default is False.

        Returns
        -------
        None.

        """
        # Get cross-hair location
        slot_offset = round(15 * self.pixels_per_mm)
        crosshair_scanline = bright(hdr[self.slot_photo_y_min - slot_offset, crosshair_search_min:crosshair_search_max], camera=True) * WHITE_EFFICACY
        x_min, x_max = slotFromScanline(crosshair_scanline, self.camera.crosshair_width, self.pixels_per_mm)
        self.crosshair_photo_x_min = x_min + crosshair_search_min + pixel_inset
        self.crosshair_photo_x_max = x_max + crosshair_search_min - pixel_inset

        # Find the offset between the photo and reference image
        ch_ref_x_min = round(hdr.shape[1] / 2 - self.camera.crosshair_left * self.pixels_per_mm)
        ch_ref_x_max = round(hdr.shape[1] / 2 - self.camera.crosshair_right * self.pixels_per_mm)
        self.x_offset = (self.crosshair_photo_x_min - ch_ref_x_min + self.crosshair_photo_x_max - ch_ref_x_max) // 2

        if plot:
            # Show photograph
            plt.imshow(to_ldr(hdr, gamma=3.5, white=0.1))
            scanline_y = self.slot_photo_y_min - slot_offset
            plt.plot([crosshair_search_min, crosshair_search_max - 1], [scanline_y, scanline_y])
            plt.plot([self.crosshair_photo_x_min, self.crosshair_photo_x_min], [0, hdr.shape[0] - 1])
            plt.plot([self.crosshair_photo_x_max, self.crosshair_photo_x_max], [0, hdr.shape[0] - 1])
            plt.show()

    def crosshairRangeX(self):
        """ The horizontal pixel range containing the vertical slot. """
        return slice(self.crosshair_photo_x_min, self.crosshair_photo_x_max)

    def findSlotX(self, luminance, photo_half_width, pixel_inset=4, min_brightness=0.001):
        """
        Find the horizontal location of the sample within the horizontal slot.

        Parameters
        ----------
        luminance : ndarray
            Horizontal scanline containing sample.
        photo_half_width : number
            Half of photo width in millimeters.
        pixel_inset : number, optional
            Distance to offset from detected edge of sample. The default is 4.
        min_brightness : number, optional
            The minimum pixel brightness that could be in part of the sample.
            The default is 0.001.

        Returns
        -------
        None.

        """
        # Get slot extents in reference image
        self.slot_ref_x_min = round((photo_half_width - self.camera.slot_left) * self.pixels_per_mm)
        self.slot_ref_x_max = round((photo_half_width + self.camera.slot_right) * self.pixels_per_mm)

        # Get slot extents in photograph
        self.slot_photo_x_min = self.slot_ref_x_min + self.x_offset
        self.slot_photo_x_max = self.slot_ref_x_max + self.x_offset

        # Get sample extents in photograph
        max_luminance_x = np.argmax(luminance[self.slot_photo_x_min:self.slot_photo_x_max]) + self.slot_photo_x_min
        sample_x_min = np.argwhere(luminance[:max_luminance_x] < min_brightness)
        sample_x_max = np.argwhere(luminance[max_luminance_x:] < min_brightness)
        if sample_x_min.size == 0 or sample_x_min[-1, 0] == sample_x_min.size - 1:
            self.sample_photo_x_min = self.slot_photo_x_min + pixel_inset
        else:
            self.sample_photo_x_min = max(sample_x_min[-1, 0], self.slot_photo_x_min) + pixel_inset
        if sample_x_max.size == 0 or sample_x_max[0, 0] == 0:
            self.sample_photo_x_max = self.slot_photo_x_max - pixel_inset
        else:
            self.sample_photo_x_max = min(sample_x_max[0, 0] + max_luminance_x, self.slot_photo_x_max) - pixel_inset

        # Get sample extents in reference image
        self.sample_ref_x_min = self.sample_photo_x_min - self.x_offset
        self.sample_ref_x_max = self.sample_photo_x_max - self.x_offset

        """
        if self.x_offset >= 0:
            self.sample_ref_x_min = self.sample_photo_x_min
            self.sample_photo_x_min += self.x_offset
            self.sample_ref_x_max = self.sample_photo_x_max - self.x_offset
        else:
            self.sample_ref_x_min = self.sample_photo_x_min - self.x_offset
            self.sample_ref_x_max = self.sample_photo_x_max
            self.sample_photo_x_max += self.x_offset
            """

    def slotRangeX(self):
        """ The horizontal pixel range of the horizontal slot. """
        return slice(self.slot_photo_x_min, self.slot_photo_x_max)

    def sampleX(self, offset=0):
        """ The horizontal pixel range containing sample within the horizontal slot. """
        return slice(self.sample_photo_x_min + offset, self.sample_photo_x_max + offset)

    def referenceX(self, offset=0):
        """ The horizontal pixel range containing the visible portion of the sample in the reference image. """
        return slice(self.sample_ref_x_min + offset, self.sample_ref_x_max + offset)

    def domainX(self):
        """ The horizontal domain of the image mapped to millimeters. """
        return np.arange(self.sample_ref_x_min, self.sample_ref_x_max) / self.pixels_per_mm

    def findCrosshairY(self, luminance, pixel_inset=4, min_brightness=0.001):
        """
        Find the vertical location of the sample within the vertical slot.

        Parameters
        ----------
        luminance : ndarray
            Horizontal scanline containing sample.
        pixel_inset : number, optional
            Distance to offset from detected edge of sample. The default is 4.
        min_brightness : number, optional
            The minimum pixel brightness that could be in part of the sample.
            The default is 0.001.

        Returns
        -------
        None.

        """
        # Get crosshair extents in reference image
        self.crosshair_ref_y_min = round((luminance.size - self.camera.crosshair_length * self.pixels_per_mm) / 2)
        self.crosshair_ref_y_max = round((luminance.size + self.camera.crosshair_length * self.pixels_per_mm) / 2)

        # Find the offset between the photo and reference image
        slot_center_y = (self.slot_photo_y_min + self.slot_photo_y_max) // 2
        self.y_offset = (self.slot_photo_y_min + self.slot_photo_y_max - luminance.shape[0]) // 2

        # Get crosshair extents in photograph
        self.crosshair_photo_y_min = self.crosshair_ref_y_min + self.y_offset
        self.crosshair_photo_y_max = self.crosshair_ref_y_max + self.y_offset

        # Get sample extents from crosshair in photograph
        sample_y_min = np.argwhere(luminance[:slot_center_y] < min_brightness)
        sample_y_max = np.argwhere(luminance[slot_center_y:] < min_brightness)
        if sample_y_min.size == 0 or sample_y_min[-1, 0] == sample_y_min.size - 1:
            self.sample_photo_y_min = self.crosshair_photo_y_min + pixel_inset
        else:
            self.sample_photo_y_min = max(sample_y_min[-1, 0], self.crosshair_photo_y_min) + pixel_inset
        if sample_y_max.size == 0 or sample_y_max[0, 0] == 0:
            self.sample_photo_y_max = self.crosshair_photo_y_max - pixel_inset
        else:
            self.sample_photo_y_max = min(sample_y_max[0, 0] + slot_center_y, self.crosshair_photo_y_max) - pixel_inset

        # Get sample extents in reference image
        self.sample_ref_y_min = self.sample_photo_y_min - self.y_offset
        self.sample_ref_y_max = self.sample_photo_y_max - self.y_offset

        """
        self.slot_offset = round(slot_center_y - luminance.size / 2)
        if self.slot_offset >= 0:
            self.reference_y_min = self.sample_photo_y_min
            self.sample_photo_y_min += self.slot_offset
            self.reference_y_max = self.sample_photo_y_max - self.slot_offset
        else:
            self.reference_y_min = self.sample_photo_y_min - self.slot_offset
            self.reference_y_max = self.sample_photo_y_max
            self.sample_photo_y_max += self.slot_offset
            """

    def crosshairRangeY(self):
        """ The vertical pixel range of the vertical slot. """
        return slice(self.crosshair_photo_y_min, self.crosshair_photo_y_max)

    def sampleY(self, offset=0):
        """ The vertical pixel range containing sample within the vertical slot. """
        return slice(self.sample_photo_y_min + offset, self.sample_photo_y_max + offset)

    def referenceY(self, offset=0):
        """ The vertical pixel range containing the visible portion of the sample in the reference image. """
        return slice(self.sample_ref_y_min + offset, self.sample_ref_y_max + offset)

    def domainY(self):
        """ The vertical domain of the image mapped to millimeters. """
        return np.arange(self.sample_ref_y_min, self.sample_ref_y_max) / self.pixels_per_mm

def diffuseCalc(diffuse_coef, specular_coef, ashik=False):
    """
    Calculate the diffuse component for Radiance

    Parameters
    ----------
    diffuse_coef : number
        Diffuse coefficient.
    specular_coef : number
        Specular coefficient.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is False.

    Returns
    -------
    diffuse : number
        The diffuse component for use in Radiance.

    """
    if ashik:
        diffuse = min(diffuse_coef, 1)
    elif specular_coef >= 1:
        diffuse = 0
    else:
        diffuse = min(diffuse_coef / (1 - specular_coef), 1)    
    return diffuse

def roughnessLookUp(row, n_rows, ashik=True):
    """
    Calculate the roughness component for Radiance

    Parameters
    ----------
    row : integer
        Best-fit row of the reference image.
    n_rows : integer
        Number of specular rows in the reference image.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is False.

    Returns
    -------
    roughness : number
        The roughness component for use in Radiance.

    """
    row_index = n_rows - 1 - row
    if ashik:
        rough_max = 10000
        roughness = rough_max * np.exp(-np.log(rough_max) * row_index / (n_rows - 1))
    else:
        roughness = row_index * 0.001
    return roughness

def fitReference(reference_image, sample, diffuse_coef=None, ashik=True, use_first_occurance=False):
    """
    Perform linear regression on a channel of the scanline.

    Parameters
    ----------
    reference_image : ndarray
        Color channel of the reference image.
    sample : ndarray
        Color channel of the scanline.
    diffuse_coef : number, optional
        Diffuse coefficient from previous calculation, or None if unknown. The
        default is None.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is True.
    use_first_occurance : boolean, optional
        Flag to use the first instance of the maximum fitting score. The
        default is False.

    Returns
    -------
    diffuse : number
        Radiance diffuse parameter.
    specular : number
        Radiance specular parameter.
    roughness : number
        Radiance roughness parameter.
    best_row : integer
        Index of the best scoring row from the reference image.
    coefs : ndarray
        Regression coefficients for each reference image row.
    scores : ndarray
        Fitting scores for each reference image row.

    """
    # Regression model
    fit_diffuse = reference_image[-1]
    if diffuse_coef is None:
        fitY = sample
    else:
        fitY = sample - fit_diffuse * diffuse_coef
    n_rows = reference_image.shape[0] - 1
    coefs = np.empty((n_rows, 2 if diffuse_coef is None else 1))
    scores = np.empty((n_rows))
    regr = lin.LinearRegression(positive=True, fit_intercept=False)

    # Find best fit for each row
    for row in range(n_rows):
        fit_specular = reference_image[row]
        if diffuse_coef is None:
            fitX = np.dstack((fit_specular, fit_diffuse))[0]
        else:
            fitX = fit_specular[:, np.newaxis]
        regr.fit(fitX, fitY)
        regr.coef_ = np.clip(regr.coef_, 0, 1) # Need to be in range
        if not ashik and diffuse_coef is None and regr.coef_.sum() > 1:
            regr.coef_[1] = 1 - regr.coef_[0]
        coefs[row] = regr.coef_
        scores[row] = regr.score(fitX, fitY)

    # Select best row
    if use_first_occurance:
        best_row = np.argmax(scores)
    else:
        best_row = round(np.flatnonzero(scores == scores.max()).mean())
    coef = coefs[best_row]

    # Diffuse calculation
    if diffuse_coef is None:
        diffuse = diffuseCalc(coef[1], coef[0], ashik=ashik)
    else:
        diffuse = diffuseCalc(diffuse_coef, coef[0], ashik=ashik)

    # Specular calculation
    specular = min(coef[0], 1)

    # Roughness calculation
    roughness = roughnessLookUp(best_row, n_rows, ashik=ashik)

    return diffuse, specular, roughness, best_row, coefs, scores

def fitReferenceDiffuseOnly(reference_image, sample, specular, rough_row):
    """
    Perform linear regression on a channel of the scanline to calculate the
    diffuse parameter only.

    Parameters
    ----------
    reference_image : ndarray
        Color channel of the reference image.
    sample : ndarray
        Color channel of the scanline.
    specular : number
        The known specular value.
    rough_row : integer
        Reference image row data corresponding to specular value.

    Returns
    -------
    diffuse : number
        Radiance diffuse parameter.
    score : number
        Fitting score.

    """
    # Regression model
    fit_diffuse = reference_image[-1, :, np.newaxis]
    fitY = sample - specular * rough_row
    regr = lin.LinearRegression(positive=True, fit_intercept=False)
    regr.fit(fit_diffuse, fitY)
    regr.coef_ = np.clip(regr.coef_, 0, 1 - specular) # Need to be in range
    score = regr.score(fit_diffuse, fitY)

    # Parameter calculation
    diffuse = diffuseCalc(regr.coef_[0], specular)
    return diffuse, score

def findBestParameterMatch(reference_image, mean_rgb, domain, plotter, markers=[], diffuse_coefs=None, ashik=True):
    """
    Find the Radiance parameters that best match the sample scanline.

    Parameters
    ----------
    reference_image : ndarray
        Color channel of the reference image.
    mean_rgb : ndarray
        Scanline from sample image.
    domain : Slice
        The x-values to plot the scanline against.
    plotter : PLotter
        Decorations for plotting.
    markers : array-like of numbers, optional
        List of markers to draw on the plots. The default is [].
    diffuse_coefs : array-like of numbers, optional
        Per-channel list of diffuse coefficients, or None if they are unknown.
        The default is None.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is True.

    Returns
    -------
    material_parameters : List
        List of Radiance material parameter values.
    record_parameters : List
        List of data to record in output.
    diffuse_coefs : List
        List of diffuse coefficients per color channel and luminance.

    """
    # Regression model
    reference_luminance = bright(reference_image)
    mean_lum = bright(mean_rgb)
    if diffuse_coefs is None:
        diffuse_l, specular_l, roughness_l, row_l, coef_l, score_l = fitReference(reference_luminance, mean_lum, ashik=ashik)
        diffuse_r, specular_r, roughness_r, row_r, coef_r, score_r = fitReference(reference_image[...,0], mean_rgb[:, 0], ashik=ashik)
        diffuse_g, specular_g, roughness_g, row_g, coef_g, score_g = fitReference(reference_image[...,1], mean_rgb[:, 1], ashik=ashik)
        diffuse_b, specular_b, roughness_b, row_b, coef_b, score_b = fitReference(reference_image[...,2], mean_rgb[:, 2], ashik=ashik)
    else:
        diffuse_l, specular_l, roughness_l, row_l, coef_l, score_l = fitReference(reference_luminance, mean_lum, diffuse_coef=diffuse_coefs[3], ashik=ashik)
        diffuse_r, specular_r, roughness_r, row_r, coef_r, score_r = fitReference(reference_image[...,0], mean_rgb[:, 0], diffuse_coef=diffuse_coefs[0], ashik=ashik)
        diffuse_g, specular_g, roughness_g, row_g, coef_g, score_g = fitReference(reference_image[...,1], mean_rgb[:, 1], diffuse_coef=diffuse_coefs[1], ashik=ashik)
        diffuse_b, specular_b, roughness_b, row_b, coef_b, score_b = fitReference(reference_image[...,2], mean_rgb[:, 2], diffuse_coef=diffuse_coefs[2], ashik=ashik)

    # Show scores
    x = np.arange(score_l.size)
    x = roughnessLookUp(x, x.size, ashik=ashik)
    plt.plot(x, score_r, color="red", label = 'Red Score')
    plt.plot(x, score_g, color="green", label = 'Green Score')
    plt.plot(x, score_b, color="blue", label = 'Blue Score')
    plt.plot(x, score_l, color="black", label = 'Luminance Score')
    if diffuse_coefs is None:
        plt.plot(x, coef_r[:,1], color="#ff8080", label = 'Red Diffuse')
        plt.plot(x, coef_g[:,1], color="#80ff80", label = 'Green Diffuse')
        plt.plot(x, coef_b[:,1], color="#8080ff", label = 'Blue Diffuse')
        plt.plot(x, coef_l[:,1], color="gray", label = 'Luminance Diffuse')
    plt.plot(x, coef_r[:,0], color="#ff8080", linestyle='--', label = 'Red Specular')
    plt.plot(x, coef_g[:,0], color="#80ff80", linestyle='--', label = 'Green Specular')
    plt.plot(x, coef_b[:,0], color="#8080ff", linestyle='--', label = 'Blue Specular')
    plt.plot(x, coef_l[:,0], color="gray", linestyle='--', label = 'Luminance Specular')
    plt.axvline(x = roughness_r, color="#ff8080", linestyle=':', linewidth=1)
    plt.axvline(x = roughness_g, color="#80ff80", linestyle=':', linewidth=1)
    plt.axvline(x = roughness_b, color="#8080ff", linestyle=':', linewidth=1)
    plt.axvline(x = roughness_l, color="gray", linestyle=':', linewidth=1)
    plt.legend(fontsize='x-small')
    plt.ylim(0, 1)
    if ashik:
        plt.xscale('log')
    plotter.plot('Score', graph=False, xlabel='Roughness', ylabel='Score')

    if diffuse_coefs is None:
        diffuse_coefs = [coef_r[row_r, 1], coef_g[row_g, 1], coef_b[row_b, 1], coef_l[row_l, 1]]
    
    # Calculate best fit
    best_fit_l = diffuse_coefs[3] * reference_luminance[-1] + coef_l[row_l, 0] * reference_luminance[row_l]
    best_fit_r = diffuse_coefs[0] * reference_image[-1,:,0] + coef_r[row_r, 0] * reference_image[row_r,:,0]
    best_fit_g = diffuse_coefs[1] * reference_image[-1,:,1] + coef_g[row_g, 0] * reference_image[row_g,:,1]
    best_fit_b = diffuse_coefs[2] * reference_image[-1,:,2] + coef_b[row_b, 0] * reference_image[row_b,:,2]

    # Show best fit
    plt.plot(domain, mean_rgb[:, 0], color="#ff8080", label = 'Sample Red')
    plt.plot(domain, mean_rgb[:, 1], color="#80ff80", label = 'Sample Green')
    plt.plot(domain, mean_rgb[:, 2], color="#8080ff", label = 'Sample Blue')
    plt.plot(domain, mean_lum, color="gray", label = 'Sample Luminance')
    plt.plot(domain, best_fit_r, color="red", label = 'Best Fit Red')
    plt.plot(domain, best_fit_g, color="green", label = 'Best Fit Green')
    plt.plot(domain, best_fit_b, color="blue", label = 'Best Fit Blue')
    plt.plot(domain, best_fit_l, color="black", label = 'Best Fit Luminance')
    for marker in markers:
        plt.axvline(x = marker, color="gray", linestyle=':')
    plotter.plot('Fit')

    # Save parameters
    record_parameters = [diffuse_l, specular_l, roughness_l, score_l[row_l], np.median(score_l),
                         diffuse_r, specular_r, roughness_r, score_r[row_r], np.median(score_r),
                         diffuse_g, specular_g, roughness_g, score_g[row_g], np.median(score_g),
                         diffuse_b, specular_b, roughness_b, score_b[row_b], np.median(score_b)]

    if ashik:
        material_parameters = [diffuse_r, diffuse_g, diffuse_b, specular_r, specular_g, specular_b, roughness_l, roughness_l]

    else:
        # Radiance definition
        diffuse_rl, score_rl = fitReferenceDiffuseOnly(reference_image[...,0], mean_rgb[:, 0], specular_l, reference_luminance[row_l])
        diffuse_gl, score_gl = fitReferenceDiffuseOnly(reference_image[...,1], mean_rgb[:, 1], specular_l, reference_luminance[row_l])
        diffuse_bl, score_bl = fitReferenceDiffuseOnly(reference_image[...,2], mean_rgb[:, 2], specular_l, reference_luminance[row_l])
        best_fit_r = (1 - specular_l) * diffuse_rl * reference_image[-1,:,0] + specular_l * reference_luminance[row_l]
        best_fit_g = (1 - specular_l) * diffuse_gl * reference_image[-1,:,1] + specular_l * reference_luminance[row_l]
        best_fit_b = (1 - specular_l) * diffuse_bl * reference_image[-1,:,2] + specular_l * reference_luminance[row_l]
        best_fit_l = bright([best_fit_r, best_fit_g, best_fit_b], camera=True)
        
        # Show best fit
        plt.plot(domain, mean_rgb[:, 0], color="#ff8080", label = 'Sample Red')
        plt.plot(domain, mean_rgb[:, 1], color="#80ff80", label = 'Sample Green')
        plt.plot(domain, mean_rgb[:, 2], color="#8080ff", label = 'Sample Blue')
        plt.plot(domain, mean_lum, color="gray", label = 'Sample Luminance')
        plt.plot(domain, best_fit_r, color="red", label = 'Best Fit Red')
        plt.plot(domain, best_fit_g, color="green", label = 'Best Fit Green')
        plt.plot(domain, best_fit_b, color="blue", label = 'Best Fit Blue')
        plt.plot(domain, best_fit_l, color="black", label = 'Best Fit Luminance')
        for marker in markers:
            plt.axvline(x = marker, color="gray", linestyle=':')
        plotter.plot('Fit2')
        
        # Save parameters
        record_parameters += [diffuse_rl, diffuse_gl, diffuse_bl, specular_l, roughness_l,
                              score_rl, score_gl, score_bl]

        material_parameters = [diffuse_rl, diffuse_gl, diffuse_bl, specular_l, roughness_l]

    return material_parameters, record_parameters, diffuse_coefs

def anisotropy(u, v):
    """
    Calculate anisotropic Radiance material parameters

    Parameters
    ----------
    u : array-like of numbers
        Radince parameters in u-direction.
    v : array-like of numbers
        Radince parameters in v-direction.

    Raises
    ------
    RuntimeError
        Bad number of parameters.

    Returns
    -------
    array-like of numbers
        Radiance parameters for anisotropic material definition.

    """
    u = np.array(u)
    v = np.array(v)
    if len(u) == 8 and len(v) == 8: # Ashikhmin-Shirley material
        ratio = np.sqrt((u[-2] + 1) / (v[-1] + 1))
        u[3:6] *= ratio
        v[3:6] /= ratio
    else: # WGDM material
        ratio = v[-1] / u[4]
        u[3] *= ratio
        v[3] /= ratio
    params = (u + v) / 2
    if len(params) == 5: # isotropic WGDM material
        params[-1] = u[-1]
        params = np.append(params, v[-1])
    elif len(params) == 6: # anisotropic WGDM material
        params[-2] = u[-2]
        params[-1] = v[-1]
    elif len(params) == 8: # Ashikhmin-Shirley material
        params[-2] = u[-2]
        params[-1] = v[-1]
    else:
        raise RuntimeError('Bad number of arguments %d for rendering material' % len(params))    
    return params.tolist()

def materialDefinition(params, name='cgi_mat'):
    """
    Generate a Radiance material parameter text string.

    Parameters
    ----------
    params : array-like of numbers
        Radiance material parameters.
    name : string, optional
        Name of the Radiance material. The default is 'cgi_mat'.

    Raises
    ------
    RuntimeError
        Bad number of parameters.

    Returns
    -------
    string
        The formatted Radiance material definiton.

    """
    if len(params) == 5: # isotropic WGDM material
        material_format = 'void plastic ' + name + '\n0\n0\n5 %g %g %g %g %g'
    elif len(params) == 6: # anisotropic WGDM material
        material_format = 'void plastic2 ' + name + '\n4 1 0 0 .\n0\n6 %g %g %g %g %g %g'
    elif len(params) == 8: # Ashikhmin-Shirley material
        material_format = 'void ashik2 ' + name + '\n4 1 0 0 .\n0\n8 %g %g %g %g %g %g %g %g'
    else:
        raise RuntimeError('Bad number of arguments %d for rendering material' % len(params))    
    return material_format % tuple(params)

class Camera:
    def __init__(self, light_height=148.6, focal_height=160.4774288, slot_left=142, slot_right=45, slot_width=10, slot_light=128.2, crosshair_left=71.4, crosshair_right=59.9, crosshair_length=80):
        """
        This is a class for the immutable properties of the camera and enclosure.

        Parameters
        ----------
        light_height : float, optional
            Height of the light diffuser above the material surface in
            millimeters. The default is 148.6.
        focal_height : float, optional
            Height of the camera's focal point above the material surface in
            millimeters, calculated using the image width and horizontal view
            angle. The default is 160.4774288.
        slot_left : float, optional
            Distance to left edge of slot from camera centerline, in
            millimeters. The default is 142.
        slot_right : float, optional
            Distance to right edge of slot from camera centerline, in
            millimeters. The default is 45.
        slot_width : float, optional
            Width of the slot in its narrow dimension in millimeters. The value
            should be less than the actual dimension for safety. The default is
            10.
        slot_light : float, optional
            Distance between the camera centerline and the center of the light
            in millimeters. The default is 128.2.
        crosshair_left : float, optional
            Distance from the left edge of the crosshair slot to the centerline
            of the camera in millimeters. The default is 71.4.
        crosshair_right : float, optional
            Distance from the right edge of the crosshair slot to the
            enterline of the camera in millimeters. The default is 59.9.
        crosshair_length : float, optional
            Length of the crosshair slot in millimeters. The default is 80.

        Returns
        -------
        None.

        """
        self.light_height = light_height # mm
        self.focal_height = focal_height # mm
        self.slot_left = slot_left # mm left of center
        self.slot_right = slot_right # mm right of center
        self.slot_width = slot_width # mm, actually 12 mm
        self.slot_light = slot_light # mm left of center
        self.crosshair_left = crosshair_left # mm left of center
        self.crosshair_right = crosshair_right # mm left of center
        self.crosshair_length = crosshair_length # mm
        self.crosshair_width = crosshair_left - crosshair_right # mm
        self.slot_highlight = slot_light * focal_height / (focal_height + light_height) # mm

class Plotter:
    def __init__(self, name, slot_width, destination=None):
        """
        This is a class for plot decorations.

        Parameters
        ----------
        name : string
            Name to appear in the plot title.
        slot_width : number
            Minimum length of the domain axis.
        destination : string, optional
            Directory path for saving plots, or None if no plots shoud not be
            saved. The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        self.plot_width = round(slot_width + 15, -1) # mm rounded up to next decade
        self.y_min = 0
        self.y_max = None
        self.destination = destination

    def plot(self, label='', graph=True, xlabel='mm', ylabel='cd/m\u00B2'):
        """
        Add decoration and save a plot.

        Parameters
        ----------
        label : string, optional
            Prefix for file name. The default is ''.
        graph : boolean, optional
            Flag for whether the plot recieved graph decorations. The default
            is True.
        xlabel : string, optional
            Label for the X-axis. The default is 'mm'.
        ylabel : string, optional
            Label for the Y-axis. The default is 'cd/m\u00B2'.

        Returns
        -------
        None.

        """
        # Plot decorations
        plt.title(self.name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if graph:
            plt.xlim(0, self.plot_width)
            if self.y_max is None:
                plt.ylim(bottom=self.y_min)
                self.y_min, self.y_max = plt.ylim()
            else:
                plt.ylim(self.y_min, self.y_max)
            plt.legend()
        if self.destination is not None:
            if label:
                file_name = '%s %s.png' % (label, self.name)
            else:
                file_name = '%s.png' % self.name
            plt.savefig(os.path.join(self.destination, file_name.strip().replace(' ', '_')))
        plt.show()

def processHDR(photo_path, reference_image_u=None, reference_image_v=None, destination=None, df=None, min_rotation=0.2, min_brightness=0.001, ashik=True):
    """
    Process and HDR image to find the sample and calculate the Radiance
    material parameters that best describe it.

    Parameters
    ----------
    photo_path : string
        Path to the HDR photograph of the sample.
    reference_image_u : ndarray, optional
        Reference image for the horizontal slot. The default is None.
    reference_image_v : ndarray, optional
        Reference image for the vertical slot. The default is None.
    destination : string, optional
        Directory path for saving plots, or None if no plots shoud not be
        saved. The default is None.
    df : DataFrame, optional
        Table of output values. The default is None.
    min_rotation : number, optional
        The minimum rotation angle to prompt a rotation operation to be
        performed on the image. The default is 0.2.
    min_brightness : number, optional
        The minimum pixel brightness that could be in part of the sample. The
        default is 0.001.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is True.

    Returns
    -------
    None.

    """
    name = os.path.basename(photo_path).replace('.hdr', '').replace('_', ' ')
    
    # Read HDR image
    hdr, header = read(photo_path)
    view_args = header['VIEW'].split()
    horizontal_angle = float(view_args[view_args.index('-vh') + 1])

    camera = Camera()

    photo_half_width = camera.focal_height * np.tan(np.deg2rad(horizontal_angle / 2))
    photo_half_height = photo_half_width * hdr.shape[0] / hdr.shape[1]
    markers = [ photo_half_width, photo_half_width - camera.slot_light, photo_half_width - camera.slot_highlight ]
    pixels_per_mm = hdr.shape[1] / (2 * photo_half_width)
    alignment = CameraAlignment(camera, pixels_per_mm)

    # Rotate if necessary
    hdr = rotateHDR(hdr, min_rotation=min_rotation, min_brightness=min_brightness)

    # Get slot location
    alignment.findSlotY(hdr)
    
    # Get mean of scanlines
    mean_rgb = hdr[alignment.slotRangeY()].mean(axis=0) * WHITE_EFFICACY
    mean_lum = bright(mean_rgb, camera=True)

    # Get cross-hair location
    alignment.findCrosshairX(hdr)

    # Get sample extents
    alignment.findSlotX(mean_lum, photo_half_width, min_brightness=min_brightness)
    geometry = [horizontal_angle, pixels_per_mm, alignment.sample_photo_x_max - alignment.sample_photo_x_min, alignment.slot_photo_y_max - alignment.slot_photo_y_min]

    plotter = Plotter(name, camera.slot_left + camera.slot_right, destination=destination)

    if alignment.slot_photo_x_min < 0:
        print("Couldn't find crosshair for %s" % name)
        return

    # Show photograph cropped to slot
    hdr_u = hdr[alignment.slotRangeY(), alignment.slotRangeX()] * WHITE_EFFICACY
    extents = [photo_half_width - camera.slot_left,
               photo_half_width + camera.slot_right,
               alignment.slot_photo_y_max / pixels_per_mm,
               alignment.slot_photo_y_min / pixels_per_mm]
    plt.imshow(to_ldr(hdr_u, gamma=3.5, white=1), extent=extents)
    for marker in markers:
        plt.plot([marker, marker], extents[2:])
    plotter.plot('Image', graph=False, xlabel='mm', ylabel=None)

    if alignment.sample_photo_x_min >= alignment.sample_photo_x_max:
        print("Couldn't find pixels for %s" % name)
        return

    # Show luminance section
    x = alignment.domainX()
    xs = alignment.sampleX()
    plt.plot(x, mean_rgb[xs, 0], color="red", label = 'Red')
    plt.plot(x, mean_rgb[xs, 1], color="green", label = 'Green')
    plt.plot(x, mean_rgb[xs, 2], color="blue", label = 'Blue')
    plt.plot(x, mean_lum[xs], color="gray", label = 'Luminance')
    for marker in markers:
        plt.axvline(x = marker, color="gray", linestyle=':')
    plotter.plot('Cut')

    # Isolate vertical slot for anisotropic calculation
    if alignment.crosshair_photo_x_min >= alignment.crosshair_photo_x_max:
        print("Couldn't find transverse pixels for %s" % name)
        return

    # Get mean of columns
    mean_rgb_v = hdr[:,alignment.crosshairRangeX()].mean(axis=1) * WHITE_EFFICACY
    mean_lum_v = bright(mean_rgb_v, camera=True)

    # Get crosshair extents
    alignment.findCrosshairY(mean_lum_v)
    geometry += [alignment.sample_photo_y_max - alignment.sample_photo_y_min, alignment.crosshair_photo_x_max - alignment.crosshair_photo_x_min]

    # Show photograph cropped to crosshair
    hdr_v = hdr[alignment.crosshairRangeY(), alignment.crosshairRangeX()] * WHITE_EFFICACY
    extents = [alignment.crosshair_photo_x_min / pixels_per_mm,
               alignment.crosshair_photo_x_max / pixels_per_mm,
               photo_half_height + camera.crosshair_length / 2,
               photo_half_height - camera.crosshair_length / 2]
    plt.imshow(to_ldr(hdr_v, gamma=3.5, white=1), extent=extents)
    plt.plot(extents[:2], [photo_half_height, photo_half_height])
    plotter.plot('Image v', graph=False, xlabel=None, ylabel='mm')

    # Show luminance section
    x = alignment.domainY()
    xs = alignment.sampleY()
    plt.plot(x, mean_rgb_v[xs, 0], color="red", label = 'Red')
    plt.plot(x, mean_rgb_v[xs, 1], color="green", label = 'Green')
    plt.plot(x, mean_rgb_v[xs, 2], color="blue", label = 'Blue')
    plt.plot(x, mean_lum_v[xs], color="gray", label = 'Luminance')
    plt.axvline(x = photo_half_height, color="gray", linestyle=':')
    plotter.plot('Cut v')

    if reference_image_u is not None:
        material_parameters, record_parameters, diffuse_coefs = findBestParameterMatch(reference_image_u[:,alignment.referenceX()], mean_rgb[alignment.sampleX()], alignment.domainX(), plotter, markers, ashik=ashik)

        if reference_image_v is not None:
            offset = int(reference_image_v.shape[1] / 2 * (1 - photo_half_height / photo_half_width))
            material_parameters_v, record_parameters_v, _ = findBestParameterMatch(reference_image_v[:,alignment.referenceY(offset)], mean_rgb_v[alignment.sampleY()], alignment.domainY(), plotter, [photo_half_height], diffuse_coefs=diffuse_coefs, ashik=ashik)
            material_parameters = anisotropy(material_parameters, material_parameters_v)
            record_parameters += record_parameters_v + material_parameters

        # Save parameters
        if df is not None:
            df.loc[name] = geometry + record_parameters

        # Print definition
        print(materialDefinition(material_parameters, name=name.replace(' ', '_')))

def run(hdr_directory, result_directory=None, ref_u_path=None, ref_v_path=None, diffuser_radiance=[0.118384, 0.109451, 0.121427], sample_count=0, save_figures=False, ashik=False, verbose_errors=False):
    """
    Calculate Radiance material parameters for a set of high dynamic range
    (HDR) images.

    Parameters
    ----------
    hdr_directory : string
        Path to the directory of input HDR images.
    result_directory : string, optional
        Path to the directory where output should be saved, or None if output
        should not be saved. If the directory does not exist, it will be
        created. The default is None.
    ref_u_path : string, optional
        Path to the horizontal reference image, or None if calculations should
        not be performed. The default is None.
    ref_v_path : string, optional
        Path to the vertical reference image, or None if the material is
        anisotropic. This input is required for the Ashikhmin-Shirley model.
        The default is None.
    diffuser_radiance : array-like of numbers, optional
        Red, Green, and Blue color channel multipliers, which are the observed
        radiance of the diffuser in each color channel. The default is
        [0.118384, 0.109451, 0.121427].
    sample_count : integer, optional
        If a non-zero value is given, the run will stop after that number of
        images have been processed. The default is 0.
    save_figures : boolean, optional
        Flag to save images generated during the run to the result_directory,
        if one is given. The default is False.
    ashik : boolean, optional
        Flag to use Ashikhmin-Shirley model. The default is True.
    verbose_errors : boolean, optional
        Flag to print detailed error messages. The default is False.

    Returns
    -------
    df_output : DataFrame
        Table of output values.

    """
    # Ensure anisotropic for Ashikhmin-Shriley model
    if ashik and not ref_v_path:
        print("Ashikhmin-Shriley model requires a transverse reference image.")

    # Output data
    geometry = ['Horizontal Angle', 'Pixels per mm', 'Slot Lengh', 'Slot Width']
    columns = [
        'Diffuse L', 'Specular L', 'Roughness L', 'Coefficient of Determination L', 'Median Score L',
        'Diffuse R', 'Specular R', 'Roughness R', 'Coefficient of Determination R', 'Median Score R',
        'Diffuse G', 'Specular G', 'Roughness G', 'Coefficient of Determination G', 'Median Score G',
        'Diffuse B', 'Specular B', 'Roughness B', 'Coefficient of Determination B', 'Median Score B']
    if not ashik:
        columns += ['Red', 'Green', 'Blue', 'Specular', 'Roughness',
            'Coefficient of Determination Red', 'Coefficient of Determination Green', 'Coefficient of Determination Blue']
    if ref_v_path:
        geometry += ['Crosshair Lengh', 'Crosshair Width']
        columns = [s + ' U' for s in columns] + [s + ' V' for s in columns]
        if ashik:
            columns += ['Red Diffuse', 'Green Diffuse', 'Blue Diffuse', 'Red Specular', 'Green Specular', 'Blue Specular', 'Roughness U', 'Roughness V']
        else:
            columns += ['Red', 'Green', 'Blue', 'Specular', 'Roughness U', 'Roughness V']
    columns = geometry + columns
    df_output = pd.DataFrame(columns=columns)

    # Output path
    if result_directory is not None and not os.path.exists(result_directory):
        os.makedirs(result_directory)
    fig_path = result_directory if save_figures else None

    # Read reference image
    reference_u = reference_image(ref_u_path, diffuser_radiance, ashik=ashik, destination=fig_path) if ref_u_path else None
    reference_v = reference_image(ref_v_path, diffuser_radiance, ashik=ashik, destination=fig_path, transverse=True, name='Transverse Reference') if ref_v_path else None

    # Add images
    count = 0
    files = os.listdir(hdr_directory)
    for file in files:
        if file.endswith('.hdr'):
            count += 1
            photo_path = os.path.join(hdr_directory, file)
            try:
                processHDR(photo_path, reference_image_u=reference_u, reference_image_v=reference_v, destination=fig_path, df=df_output, ashik=ashik)
            except Exception as e:
                if verbose_errors:
                    traceback.print_exc()
                else:
                    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
            if sample_count > 0 and count >= sample_count:
                break

    # Save outputs
    result_name = 'ashik' if ashik else 'wgmd'
    if ref_v_path:
        result_name += '2'
    if sample_count > 0:
        result_name += '_%d' % sample_count
    if result_directory is not None and not df_output.empty:
        df_output.to_csv(os.path.join(result_directory, result_name + '.csv'))

    return df_output
