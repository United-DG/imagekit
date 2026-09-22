"""Background remover — core (GUI-free) and GUI layers.

The `rmbg` package is split so the imaging pipeline can be used headlessly:

    rmbg.config    constants, the model registry, output formats
    rmbg.params    the Params settings object
    rmbg.imaging   load / trim / upscale / encode
    rmbg.mask      alpha post-processing (the quality levers)
    rmbg.engine    rembg session management and inference
    rmbg.pipeline  refine / compose / render / export
    rmbg.edit      manual touch-up layer (brush, click-to-remove, undo)
    rmbg.batch     folder and ZIP processing
    rmbg.cli       headless command line

Nothing under `rmbg/` outside `rmbg.gui` imports a GUI toolkit.
"""

__version__ = "2.0.0"
