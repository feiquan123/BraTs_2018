import os, re, math
import glob
import nibabel as nib
import imageio
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def unzip(src: str, tag: str):
    """
    .nill.gz -> .png
    :param src:
    :param tag:
    :return:
    """
    path = os.path.join(tag, src[:-7])
    if not os.path.exists(path):
        os.makedirs(path)

    img = nib.load(src).get_fdata()
    [imageio.imwrite(os.path.join(path, "%d.png" % i), img[i].astype(np.uint8)) for i in range(img.shape[-1])]


def saveimage(data, idx: int, path: str):
    plt.axis('off')
    plt.style.use("ggplot")
    plt.imshow(data, cmap='Greys_r')
    plt.imshow(data)
    plt.savefig(os.path.join(path, "%d.png" % idx))
    plt.close()
    print(os.path.join(path, "%d.png" % idx))

def unzip2(src: str, tag: str):
    path = os.path.join(tag, src[:-7])
    if not os.path.exists(path):
        os.makedirs(path)

    img = sitk.GetArrayFromImage(sitk.ReadImage(src))
    [saveimage((img[i] == 2).astype(np.uint8), i, path) for i in range(img.shape[0])]


# convert to image
def convert(src: str, tag: str, size: int = 0, unzip=unzip):
    """
    unzip .nii.gz to png
    :param src: src path
    :param tag: tag path
    :param len: when len == 0 , unzip all
    :return:
    """
    t1 = glob.glob(f'{src}/*GG/*/*t1.nii.gz')
    t2 = glob.glob(f'{src}/*GG/*/*t2.nii.gz')
    flair = glob.glob(f'{src}/*GG/*/*flair.nii.gz')
    t1ce = glob.glob(f'{src}/*GG/*/*t1ce.nii.gz')
    seg = glob.glob(f'{src}/*GG/*/*seg.nii.gz')  # Ground Truth
    pat = re.compile('.*_(\w*)\.nii\.gz')

    data_paths = [{
        pat.findall(item)[0]: item
        for item in items
    }
        for items in list(zip(t1, t2, t1ce, flair, seg))]

    if not size:
        size = len(data_paths)
    total = len(data_paths[:size])
    step = 25 / total

    for i, imgs in enumerate(data_paths[:size]):
        try:
            [unzip(imgs[m], tag) for m in ['t1', 't2', 't1ce', 'flair', 'seg']]
            print('\r\n' + f'Progress: '
                           f"[{'=' * int((i + 1) * step) + ' ' * (24 - int((i + 1) * step))}]"
                           f"({math.ceil((i + 1) * 100 / (total))} %)" + '\r\n',
                  end=''
                  )
        except Exception as e:
            print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
            continue


if __name__ == '__main__':
    # unzip("HGG/Brats18_CBICA_AOO_1/Brats18_CBICA_AOO_1_seg.nii.gz", "target")
    convert(".", "target", size=4,unzip= unzip)

    # unzip2("HGG/Brats18_CBICA_AOO_1/Brats18_CBICA_AOO_1_seg.nii.gz", "target")
    # convert(".", "target", size=4,unzip= unzip2)
