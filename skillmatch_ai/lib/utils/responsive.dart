// lib/utils/responsive.dart
//
// Central place for breakpoints + small helpers so every screen adapts
// consistently across phone, tablet, and web/desktop without each screen
// re-inventing its own thresholds.

import 'package:flutter/material.dart';

/// Width thresholds used everywhere in the app.
/// < tablet        -> phone layout
/// tablet..desktop  -> tablet layout
/// >= desktop       -> desktop / web layout
class Breakpoints {
  static const double tablet = 600;
  static const double desktop = 1024;
}

extension ResponsiveContext on BuildContext {
  double get screenWidth => MediaQuery.of(this).size.width;

  bool get isMobile => screenWidth < Breakpoints.tablet;
  bool get isTablet =>
      screenWidth >= Breakpoints.tablet && screenWidth < Breakpoints.desktop;
  bool get isDesktop => screenWidth >= Breakpoints.desktop;

  /// True once there's enough width to show a side navigation rail
  /// instead of a bottom nav bar (tablet landscape and up).
  bool get useRailNav => screenWidth >= Breakpoints.tablet;
}

/// Picks a value based on the current width. Falls back gracefully:
/// desktop -> tablet -> mobile, tablet -> mobile.
T responsiveValue<T>(
  BuildContext context, {
  required T mobile,
  T? tablet,
  T? desktop,
}) {
  if (context.isDesktop) return desktop ?? tablet ?? mobile;
  if (context.isTablet) return tablet ?? mobile;
  return mobile;
}

/// Wraps a screen's scrollable content so it stays a comfortable, readable
/// column on tablet/desktop/web (centered, capped width) while still
/// filling the screen edge-to-edge on phones.
///
/// Usage: wrap the widget you'd otherwise pass straight to
/// SingleChildScrollView/CustomScrollView's child with this.
class ResponsiveCenter extends StatelessWidget {
  final Widget child;

  /// Max width of the content column once the screen is wide enough.
  final double maxWidth;

  const ResponsiveCenter({
    super.key,
    required this.child,
    this.maxWidth = 720,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: maxWidth),
        child: child,
      ),
    );
  }
}

/// Same idea as [ResponsiveCenter] but for use directly as a `body:`
/// (i.e. wraps something that already fills/scrolls its own height, like a
/// SingleChildScrollView or CustomScrollView) — just caps + centers width.
class ResponsiveBody extends StatelessWidget {
  final Widget child;
  final double maxWidth;

  const ResponsiveBody({
    super.key,
    required this.child,
    this.maxWidth = 720,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: maxWidth),
        child: child,
      ),
    );
  }
}

/// Returns a horizontal gutter that grows slightly on larger screens so
/// content doesn't feel cramped against the edges when maxWidth hasn't
/// kicked in yet (e.g. a tablet in the 600-720 range).
double responsiveGutter(BuildContext context, double basePadding) {
  if (context.isDesktop) return basePadding + 16;
  if (context.isTablet) return basePadding + 8;
  return basePadding;
}
