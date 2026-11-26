/**
 * Ray pointers currently pointing at a VR UI element. While any is present the control hand's trigger value is
 * frozen, so pressing a menu button does not open/close the gripper (or drop a held object).
 *
 * Only *ray* pointers count: each controller also carries a touch pointer (10 cm hover radius) and a grab
 * pointer (7 cm). The hand menu floats right above the other controller, so those would otherwise "hover"
 * it permanently and freeze the trigger forever.
 */
const rays = new Set<number>()

export const uiHover = {
  get active() {
    return rays.size > 0
  },
  enter(pointerId: number, pointerType: string) {
    if (pointerType === 'ray') rays.add(pointerId)
  },
  leave(pointerId: number) {
    rays.delete(pointerId)
  },
  clear() {
    rays.clear()
  },
}
