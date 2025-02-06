# squid_vision_paper_draft

# What is this?
* Lorenz did a fantastic job in sharing codes when he published Nature 2023 paper (See https://gitlab.mpcdf.mpg.de/mpibr/laur/inter-hemispheric-rem). I want to follow his approach in our paper.
* The aim is to post all codes and jupyter notebooks that was used to generate figures here.
    * Assume that you will save all outputs in `figures/outs`
    * I wrote .gitignore so that `fiures/outs` will not be commited to git.
    * As an example, see `figures/20240904_S1504_LCD_calibration.ipynb`
* No need to put all codes that were used for preprocessing (e.g. motion correction, spike sorting, etc). Only the scripts that generated the final figure should be posted here.
* Ideally I want to assemble preprocessing codes in one place, but let's discuss it later.
* I regard this as a draft repository. At the point of publication I plan to make another clean repository with all commit history erased. So please feel free to upload dirty codes and notebooks that might contain some personal information (like password for SSH? just kidding...).
* Let us also put the singularity images or conda environment files to run the code here.
    * As an example, see `/singularity/CaImAn/READMD.md`

# Guidelines for generating plots using matplotlib

To ensure that `matplotlib` generates PDF files compatible with Adobe Illustrator (avoiding common issues like font recognition errors), it’s important to configure it properly. 

Below is a set of commands that have worked reliably in my experience. Add these lines at the beginning of your script to produce PDFs that import beautifully into Illustrator.
 If you encounter any issues, feel free to reach out.

```python
from matplotlib import font_manager
import matplotlib as mpl

font_manager.fontManager.addfont('/bucket/ReiterU/common/fonts/arial.ttf')
mpl.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': "Arial",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'xtick.major.size': 2,
    'xtick.minor.size': 2,
    'xtick.minor.width': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.width': 0.5,
    'ytick.major.size': 2,
    'ytick.minor.size': 2,
})
```

Explanation of the Code
* Adding the Arial Font: The Arial TTF file is located at `/bucket/ReiterU/common/fonts/arial.ttf`. This path is provided to `matplotlib.font_manager` to ensure that Matplotlib can use the font.
* Setting Default Font and Size: The configuration specifies Arial as the default font and sets the font size to **7**.
* Ensuring Illustrator Compatibility: The pdf.fonttype and ps.fonttype are set to 42, which ensures the fonts are properly embedded and recognized when the PDF is imported into Illustrator.
* Adjusting the tick and bounding box line widths: Here it uses **0.5** points, which should look nice on figures.
* Even with this approach, I still have issues with greek letters, which are exported with default Dejavu-Sans font. It can be fixed, but the solution was convoluted and I want to stay away from it...
