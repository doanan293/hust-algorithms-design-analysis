from models.channel import LinkChannel, all_uav_links
from models.paths import radio_links


def constant_channels(scenario, candidates, snr_per_hz, overrides=None):
    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    channels = {link: LinkChannel.constant(scenario.time.num_slots, snr_per_hz) for link in links}
    channels.update(overrides or {})
    return channels
