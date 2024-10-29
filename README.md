# Clothes-Virtual-Try-On

## Resources:
* Example guide: https://www.youtube.com/watch?v=C94pTaKoLbU

* Possible model for virtual try-on: https://github.com/levihsu/OOTDiffusion
** Overall, the performance is good on upper, lower and full body with clothes and dress. However, there are some limitations:
"First, since our models are trained on paired human and garment
images, it may fail to get perfect results for cross-category virtual try-on, e.g., to
put a T-shirt on a woman in a long dress, or to let a man in pants wear a skirt.
This issue can be partially solved in the future by collecting datasets of each
person wearing different clothes in the same pose. Another limitation is that
some details in the original human image might be altered after virtual try-on,
such as muscles, watches or tattoos, etc. The reason is that the relevant body
area is masked and repainted by the diffusion model. Thus more practical pre-
and post-processing methods are required for addressing such problems."

* Simple guide to build a Full-stack Generative AI App: https://xiaolishen.medium.com/a-fullstack-text-to-image-generative-ai-app-you-can-build-in-an-afternoon-31990657344b
