// Resolve a packed (resource slot, byte offset) entry against ordinary Metal
// buffer arguments. Unlike a raw GPU address, each source allocation is
// visible to Metal's encoder resource tracker.
inline const device uint8_t *glm53_bound_resource_base(
    uint slot,
    const device uint8_t *resource_00,
    const device uint8_t *resource_01,
    const device uint8_t *resource_02,
    const device uint8_t *resource_03,
    const device uint8_t *resource_04,
    const device uint8_t *resource_05,
    const device uint8_t *resource_06,
    const device uint8_t *resource_07,
    const device uint8_t *resource_08,
    const device uint8_t *resource_09,
    const device uint8_t *resource_10,
    const device uint8_t *resource_11,
    const device uint8_t *resource_12,
    const device uint8_t *resource_13,
    const device uint8_t *resource_14,
    const device uint8_t *resource_15) {
    switch (slot) {
        case 0: return resource_00;
        case 1: return resource_01;
        case 2: return resource_02;
        case 3: return resource_03;
        case 4: return resource_04;
        case 5: return resource_05;
        case 6: return resource_06;
        case 7: return resource_07;
        case 8: return resource_08;
        case 9: return resource_09;
        case 10: return resource_10;
        case 11: return resource_11;
        case 12: return resource_12;
        case 13: return resource_13;
        case 14: return resource_14;
        case 15: return resource_15;
        default: return nullptr;
    }
}

inline const device uint8_t *glm53_mapped_resource(
    ulong entry,
    const device uint8_t *resource_00,
    const device uint8_t *resource_01,
    const device uint8_t *resource_02,
    const device uint8_t *resource_03,
    const device uint8_t *resource_04,
    const device uint8_t *resource_05,
    const device uint8_t *resource_06,
    const device uint8_t *resource_07,
    const device uint8_t *resource_08,
    const device uint8_t *resource_09,
    const device uint8_t *resource_10,
    const device uint8_t *resource_11,
    const device uint8_t *resource_12,
    const device uint8_t *resource_13,
    const device uint8_t *resource_14,
    const device uint8_t *resource_15) {
    const uint slot = uint(entry >> 56);
    const ulong offset = entry & 0x00fffffffffffffful;
    const device uint8_t *base = glm53_bound_resource_base(
        slot,
        resource_00,
        resource_01,
        resource_02,
        resource_03,
        resource_04,
        resource_05,
        resource_06,
        resource_07,
        resource_08,
        resource_09,
        resource_10,
        resource_11,
        resource_12,
        resource_13,
        resource_14,
        resource_15);
    return base == nullptr ? nullptr : base + offset;
}
