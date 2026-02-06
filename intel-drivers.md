There are two software packages that need to be installed simultaneously
for optimal video encoding support for Jellyfin and Plex. Ignore the
version numbers:

firmware-linux-nonfree/testing,now 20250410-2 all [installed]
  Binary firmware for various drivers in the Linux kernel (metapackage)

While they offer the video encoding interfacesi (VAAPI) for Jellyfin and Plex,
they also break xserver/xorg/Gnome. This log file saves errors during xserver
startup:
/var/lib/gdm3/.local/share/xorg/Xorg.1.log

This file will end with text like:
```
[   656.834] (II) LoadModule: "vesa"
[   656.834] (II) Loading /usr/lib/xorg/modules/drivers/vesa_drv.so
[   656.834] (II) Module vesa: vendor="X.Org Foundation"
[   656.834]    compiled for 1.21.1.15, module version = 2.6.0
[   656.834]    Module class: X.Org Video Driver
[   656.834]    ABI class: X.Org Video Driver, version 25.2
[   656.834] (II) modesetting: Driver for Modesetting Kernel Drivers: kms
[   656.834] (II) FBDEV: driver for framebuffer: fbdev
[   656.834] (II) VESA: driver for VESA chipsets: vesa
[   656.834] (WW) xf86OpenConsole: VT_ACTIVATE failed: Operation not permitted
[   656.834] (EE)
Fatal server error:
[   656.834] (EE) xf86OpenConsole: Switching VT failed
[   656.834] (EE)
[   656.834] (EE)
Please consult the The X.Org Foundation support
         at http://wiki.x.org
 for help.
[   656.834] (EE) Please also check the log file at "/var/lib/gdm3/.local/share/xorg/Xorg.1.log" for additional information.
[   656.834] (EE)
[   656.834] (WW) xf86CloseConsole: KDSETMODE failed: Operation not permitted
[   656.834] (WW) xf86CloseConsole: VT_SETMODE failed: Operation not permitted
[   656.873] (EE) Server terminated with error (1). Closing log file.
```

This failure is a signature of using nonfree drivers from Intel.

### What follows is SPECULATION ###

Restoring video capability requires reverting back to xserver-compatible
drivers, which unfortunately do not offer the latest GPU video encoding
interfaces like HDR or AV1 encoding. The packages that work with
xserver/xorg/Gnome are:

xserver-xorg-video-intel/testing 2:2.99.917+git20210115-1 amd64
  X.Org X server -- Intel i8xx, i9xx display driver

firmware-linux-free/testing,now 20241210-2 all [installed]
  Binary firmware for various drivers in the Linux kernel

firmware-intel-graphics/testing 20250410-2 all
  Binary firmware for Intel iGPUs and IPUs

However, as mentioned previously, they do not support the VAAPI GPU hardware
encoding interface needed for Jellyfin and Plex. While it is possible to have
xserver-xorg-video-intel installed at the same time as
intel-media-va-driver-non-free, but Linux will seem to prefer and load
xserver-xorg-video-intel; breaking hardware encoding support. So, for the sake
of explicit driver selection, it is recommended to only have one of these two
drivers installed at any point in time.

This information is estimated from my experience on Debian Trixie, and from
reading documentation here:
https://wiki.archlinux.org/title/Hardware_video_acceleration#Intel
https://wiki.archlinux.org/title/Intel_graphics

The Arch documentation requires some interpretation of Arch package names to
Debian package names, which may not be the same or may even be a completely
different package of resources from one distro to another.

Additionally:
- Do not install any xf86 packages. They are not recommended in any distrubtion.
- I do not know the difference between the firmware-intel-graphics package and
  the intel firmwares that come in firmware-linux-nonfree.
- I do not know the difference between the intel-media-va-driver and
  intel-media-va-driver-non-free packages.
- It might be worth trying intel-media-va-driver and firmware-intel-graphics to
  see if a solution exists with VAAPI and working xserver.
- The firmwares of some kind are necessary.
  Graphics micro (μ) Controller (GuC) and
  HEVC/H.265 micro (µ) Controller (HuC) are included specifically in firmware
  files.


Just did some experimentation and I've learned that, to make Gnome graphics
function: the following package combiation MUST be used:
firmware-linux-free  +  firmware-intel-graphics

The following packages must be uninstalled:
firmware-linux-nonfree

These drivers work with Gnome in both situations:

intel-media-va-driver-non-free/testing,now 25.2.3+ds1-1 amd64 [installed]
  VAAPI driver for the Intel GEN8+ Graphics family

intel-media-va-driver/testing 25.2.3+dfsg1-1 amd64
  VAAPI driver for the Intel GEN8+ Graphics family


Both Plex and Gnome appear to be working with:

xserver-xorg-video-intel/testing 2:2.99.917+git20210115-1 amd64
  X.Org X server -- Intel i8xx, i9xx display driver

firmware-linux-free/testing,now 20241210-2 all [installed]
  Binary firmware for various drivers in the Linux kernel

firmware-intel-graphics/testing 20250410-2 all
  Binary firmware for Intel iGPUs and IPUs

intel-media-va-driver-non-free/testing,now 25.2.3+ds1-1 amd64 [installed]
  VAAPI driver for the Intel GEN8+ Graphics family

- No i915 drivers
- Np xf86 drivers
- This can be verified during transcoding with intel_gpu_top
