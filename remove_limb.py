def black_mask(s_map, value, limb_offset=0.99):

    coords = all_coordinates_from_map(s_map)
    radial_distance = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs.value
    radial_distance = radial_distance.value

    s_map_data = s_map.data.copy()
    s_map_data = s_map.data.astype(np.float32)

    s_map_data[radial_distance >= limb_offset] = value

    solar_disk = s_map_data[radial_distance < limb_offset]

    return s_map_data, solar_disk.mean(), solar_disk.std()