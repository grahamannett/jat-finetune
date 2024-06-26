import warnings

import datasets


def get_jat_config_names() -> list[str]:
    """Get the list of available JAT dataset config names.
    WARNING: this is insanely slow.  might be wise to cache this (or just copy the strs) to a file
    """
    warnings.simplefilter("always", ResourceWarning)
    warnings.warn(
        "this is insanely slow, prefer to use `JAT_DATASET_CONFIG_NAMES` instead.",
        category=ResourceWarning,
        stacklevel=1,
    )
    warnings.simplefilter("default", ResourceWarning)
    return datasets.get_dataset_config_names("jat-project/jat-dataset")


JAT_DATASET_CONFIG_NAMES = [
    "atari-alien",
    "atari-amidar",
    "atari-assault",
    "atari-asterix",
    "atari-asteroids",
    "atari-atlantis",
    "atari-bankheist",
    "atari-battlezone",
    "atari-beamrider",
    "atari-berzerk",
    "atari-bowling",
    "atari-boxing",
    "atari-breakout",
    "atari-centipede",
    "atari-choppercommand",
    "atari-crazyclimber",
    "atari-defender",
    "atari-demonattack",
    "atari-doubledunk",
    "atari-enduro",
    "atari-fishingderby",
    "atari-freeway",
    "atari-frostbite",
    "atari-gopher",
    "atari-gravitar",
    "atari-hero",
    "atari-icehockey",
    "atari-jamesbond",
    "atari-kangaroo",
    "atari-krull",
    "atari-kungfumaster",
    "atari-montezumarevenge",
    "atari-mspacman",
    "atari-namethisgame",
    "atari-phoenix",
    "atari-pitfall",
    "atari-pong",
    "atari-privateeye",
    "atari-qbert",
    "atari-riverraid",
    "atari-roadrunner",
    "atari-robotank",
    "atari-seaquest",
    "atari-skiing",
    "atari-solaris",
    "atari-spaceinvaders",
    "atari-stargunner",
    "atari-surround",
    "atari-tennis",
    "atari-timepilot",
    "atari-tutankham",
    "atari-upndown",
    "atari-venture",
    "atari-videopinball",
    "atari-wizardofwor",
    "atari-yarsrevenge",
    "atari-zaxxon",
    "babyai-action-obj-door",
    "babyai-blocked-unlock-pickup",
    "babyai-boss-level",
    "babyai-boss-level-no-unlock",
    "babyai-find-obj-s5",
    "babyai-go-to",
    "babyai-go-to-door",
    "babyai-go-to-imp-unlock",
    "babyai-go-to-local",
    "babyai-go-to-obj",
    "babyai-go-to-obj-door",
    "babyai-go-to-red-ball",
    "babyai-go-to-red-ball-grey",
    "babyai-go-to-red-ball-no-dists",
    "babyai-go-to-red-blue-ball",
    "babyai-go-to-seq",
    "babyai-key-corridor",
    "babyai-mini-boss-level",
    "babyai-move-two-across-s8n9",
    "babyai-one-room-s8",
    "babyai-open",
    "babyai-open-door",
    "babyai-open-doors-order-n4",
    "babyai-open-red-door",
    "babyai-open-two-doors",
    "babyai-pickup",
    "babyai-pickup-above",
    "babyai-pickup-dist",
    "babyai-pickup-loc",
    "babyai-put-next",
    "babyai-put-next-local",
    "babyai-synth",
    "babyai-synth-loc",
    "babyai-synth-seq",
    "babyai-unblock-pickup",
    "babyai-unlock",
    "babyai-unlock-local",
    "babyai-unlock-pickup",
    "babyai-unlock-to-unlock",
    "conceptual-captions",
    "metaworld-assembly",
    "metaworld-basketball",
    "metaworld-bin-picking",
    "metaworld-box-close",
    "metaworld-button-press",
    "metaworld-button-press-topdown",
    "metaworld-button-press-topdown-wall",
    "metaworld-button-press-wall",
    "metaworld-coffee-button",
    "metaworld-coffee-pull",
    "metaworld-coffee-push",
    "metaworld-dial-turn",
    "metaworld-disassemble",
    "metaworld-door-close",
    "metaworld-door-lock",
    "metaworld-door-open",
    "metaworld-door-unlock",
    "metaworld-drawer-close",
    "metaworld-drawer-open",
    "metaworld-faucet-close",
    "metaworld-faucet-open",
    "metaworld-hammer",
    "metaworld-hand-insert",
    "metaworld-handle-press",
    "metaworld-handle-press-side",
    "metaworld-handle-pull",
    "metaworld-handle-pull-side",
    "metaworld-lever-pull",
    "metaworld-peg-insert-side",
    "metaworld-peg-unplug-side",
    "metaworld-pick-out-of-hole",
    "metaworld-pick-place",
    "metaworld-pick-place-wall",
    "metaworld-plate-slide",
    "metaworld-plate-slide-back",
    "metaworld-plate-slide-back-side",
    "metaworld-plate-slide-side",
    "metaworld-push",
    "metaworld-push-back",
    "metaworld-push-wall",
    "metaworld-reach",
    "metaworld-reach-wall",
    "metaworld-shelf-place",
    "metaworld-soccer",
    "metaworld-stick-pull",
    "metaworld-stick-push",
    "metaworld-sweep",
    "metaworld-sweep-into",
    "metaworld-window-close",
    "metaworld-window-open",
    "mujoco-ant",
    "mujoco-doublependulum",
    "mujoco-halfcheetah",
    "mujoco-hopper",
    "mujoco-humanoid",
    "mujoco-pendulum",
    "mujoco-pusher",
    "mujoco-reacher",
    "mujoco-standup",
    "mujoco-swimmer",
    "mujoco-walker",
    "ok-vqa",
    "oscar",
    "wikipedia",
]
