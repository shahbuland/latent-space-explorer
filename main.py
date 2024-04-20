from latent_space_explorer import GameConfig, LatentSpaceExplorer

if __name__ == "__main__":
    config = GameConfig(
        call_every = 100
    )

    explorer = LatentSpaceExplorer(config)

    explorer.set_prompts(
        [
            "A photo of a cat",
            "A space-aged ferrari",
            "artwork of the titanic hitting an iceberg",
            "a photo of a dog"
        ]
    )

    while True:
        explorer.update()