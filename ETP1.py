#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import urllib

def get_filename(url):
    """
    Parses filename from given url
    """
    if url.find('/'):
        return url.rsplit('/', 1)[1]

# Filepaths
outdir = r"data"

# File locations
url_list = ["https://github.com/Automating-GIS-processes/CSC18/raw/master/data/Helsinki_masked_p188r018_7t20020529_z34__LV-FIN.tif"]

# Create folder if it does no exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Download files
for url in url_list:
    # Parse filename
    fname = get_filename(url)
    outfp = os.path.join(outdir, fname)
    # Download the file if it does not exist already
    if not os.path.exists(outfp):
        print("Downloading", fname)
        r = urllib.request.urlretrieve(url, outfp)


# In[3]:


import rasterio
import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = "D:\IGIS\IGIS SEM 6\GIS PROGRAMMING LAB"
fp = os.path.join(data_dir, "Helsinki_masked_p188r018_7t20020529_z34__LV-FIN.tif")

# Open the file:
raster = rasterio.open(fp)

# Check type of the variable 'raster'
type(raster)


# In[4]:


raster.crs


# In[5]:


raster.transform


# In[6]:


# Read the raster band as separate variable
band1 = raster.read(1)

# Check type of the variable 'band'
print(type(band1))

# Data type of the values
print(band1.dtype)


# In[7]:


# Read all bands
array = raster.read()

# Calculate statistics for each band
stats = []
for band in array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()})

# Show stats for each channel
stats


# In[9]:


import rasterio
from rasterio.plot import show
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = "D:\IGIS\IGIS SEM 6\GIS PROGRAMMING LAB"

# Filepath
fp = os.path.join(data_dir, "Helsinki_masked_p188r018_7t20020529_z34__LV-FIN.tif")

# Open the file:
raster = rasterio.open(fp)

# Plot band 1
show((raster, 1))


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Initialize subplots 
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(10, 4), sharey=True)

# Plot Red, Green and Blue (rgb)
show((raster, 4), cmap='Reds', ax=ax1)
show((raster, 3), cmap='Greens', ax=ax2)
show((raster, 1), cmap='Blues', ax=ax3)

# Add titles
ax1.set_title("Red")
ax2.set_title("Green")
ax3.set_title("Blue")


# In[11]:


# Read the grid values into numpy arrays
red = raster.read(3)
green = raster.read(2)
blue = raster.read(1)

# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)

print("Normalized bands")
print(redn.min(), '-', redn.max(), 'mean:', redn.mean())
print(greenn.min(), '-', greenn.max(), 'mean:', greenn.mean())
print(bluen.min(), '-', bluen.max(), 'mean:', bluen.mean())


# In[12]:


# Create RGB natural color composite
rgb = np.dstack((redn, greenn, bluen))

# Let's see how our color composite looks like
plt.imshow(rgb)


# In[13]:


# Read the grid values into numpy arrays
nir = raster.read(4)
red = raster.read(3)
green = raster.read(2)

# Normalize the values using the function that we defined earlier
nirn = normalize(nir)
redn = normalize(red)
greenn = normalize(green)

# Create the composite by stacking
nrg = np.dstack((nirn, redn, greenn))

# Let's see how our color composite looks like
plt.imshow(nrg)


# In[14]:


from rasterio.plot import show_hist
    
show_hist(raster, bins=50, lw=0.0, stacked=False, alpha=0.3,
      histtype='stepfilled', title="Histogram")


# In[5]:


import rasterio
import numpy as np
from rasterio.plot import show
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = "D:\IGIS\IGIS SEM 6\GIS PROGRAMMING LAB"

# Filepath
fp = os.path.join(data_dir, "Helsinki_masked_p188r018_7t20020529_z34__LV-FIN.tif")

# Open the raster file in read mode
raster = rasterio.open(fp)


# In[6]:


# Read red channel (channel number 3)
red = raster.read(3)
# Read NIR channel (channel number 4)
nir = raster.read(4)

# Calculate some stats to check the data
print(red.mean())
print(nir.mean())
print(type(nir))

# Visualize
show(nir, cmap='terrain')


# In[7]:


# Convert to floats
red = red.astype('f4')
nir = nir.astype('f4')
nir


# In[8]:


np.seterr(divide='ignore', invalid='ignore')


# In[9]:


# Calculate NDVI using numpy arrays
ndvi = (nir - red) / (nir + red)


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Plot the NDVI
plt.imshow(ndvi, cmap='terrain_r')
# Add colorbar to show the index
plt.colorbar()


# In[13]:


import rasterio
import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = r"F:\IGIS\Lahoreee\2000dn"
fp = os.path.join(data_dir, "allbands.tif")

# Open the file:
raster = rasterio.open(fp)

# Check type of the variable 'raster'
type(raster)


# In[14]:


# Projection
raster.crs


# In[15]:


# Affine transform (how raster is scaled, rotated, skewed, and/or translated)
raster.transform


# In[16]:


# Dimensions
print(raster.width)
print(raster.height)


# In[17]:


# Number of bands
raster.count


# In[18]:


# Bounds of the file
raster.bounds


# In[19]:


# Driver (data format)
raster.driver


# In[20]:


# No data values for all channels
raster.nodatavals


# In[21]:


# All Metadata for the whole raster dataset
raster.meta


# In[22]:


# Read the raster band as separate variable
band1 = raster.read(1)

# Check type of the variable 'band'
print(type(band1))

# Data type of the values
print(band1.dtype)


# In[1]:


# Read all bands
array = raster.read()

# Calculate statistics for each band
stats = []
for band in array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()})

# Show stats for each channel
stats


# In[2]:


import rasterio
import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = r"F:\IGIS\Lahoreee\2000dn"
fp = os.path.join(data_dir, "allbands.tif")

# Open the file:
raster = rasterio.open(fp)

# Check type of the variable 'raster'
type(raster)


# In[3]:


# Affine transform (how raster is scaled, rotated, skewed, and/or translated)
raster.transform


# In[4]:


# No data values for all channels
raster.nodatavals


# In[5]:


# Read the raster band as separate variable
band1 = raster.read(1)

# Check type of the variable 'band'
print(type(band1))

# Data type of the values
print(band1.dtype)


# In[6]:


# Read all bands
array = raster.read()

# Calculate statistics for each band
stats = []
for band in array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()})

# Show stats for each channel
stats


# In[11]:


import rasterio
from rasterio.plot import show
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Data dir
data_dir = r"F:\IGIS\Lahoreee\2000dn"

# Filepath
fp = os.path.join(data_dir, "allbands.tif")

# Open the file:
raster = rasterio.open(fp)

# Plot band 1
show((raster, 1))


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Initialize subplots 
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(10, 4), sharey=True)

# Plot Red, Green and Blue (rgb)
show((raster, 4), cmap='Reds', ax=ax1)
show((raster, 3), cmap='Greens', ax=ax2)
show((raster, 1), cmap='Blues', ax=ax3)

# Add titles
ax1.set_title("Red")
ax2.set_title("Green")
ax3.set_title("Blue")


# In[13]:


# Read the grid values into numpy arrays
red = raster.read(3)
green = raster.read(2)
blue = raster.read(1)

# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)

print("Normalized bands")
print(redn.min(), '-', redn.max(), 'mean:', redn.mean())
print(greenn.min(), '-', greenn.max(), 'mean:', greenn.mean())
print(bluen.min(), '-', bluen.max(), 'mean:', bluen.mean())


# In[14]:


# Create RGB natural color composite
rgb = np.dstack((redn, greenn, bluen))

# Let's see how our color composite looks like
plt.imshow(rgb)


# In[15]:


# Read the grid values into numpy arrays
nir = raster.read(4)
red = raster.read(3)
green = raster.read(2)

# Normalize the values using the function that we defined earlier
nirn = normalize(nir)
redn = normalize(red)
greenn = normalize(green)

# Create the composite by stacking
nrg = np.dstack((nirn, redn, greenn))

# Let's see how our color composite looks like
plt.imshow(nrg)


# In[1]:


from rasterio.plot import show_hist
    
show_hist(raster, bins=50, lw=0.0, stacked=False, alpha=0.3,
      histtype='stepfilled', title="Histogram")


# In[ ]:




