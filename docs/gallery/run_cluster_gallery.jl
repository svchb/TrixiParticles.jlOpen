#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "GalleryPipeline.jl"))

exit(GalleryPipeline.main(ARGS))
