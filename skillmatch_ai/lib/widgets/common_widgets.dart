// lib/widgets/common_widgets.dart
// Neo-brutalist shared components: thick outlines, hard offset shadows,
// flat color blocks. These are reused across every screen so changing them
// here cascades the new look app-wide.

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';

/// Bento-tile stat card — flat color block, thick outline, hard shadow.
class StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color valueColor;
  final IconData icon;
  final Color? blockColor;

  const StatCard({
    super.key,
    required this.label,
    required this.value,
    this.valueColor = AppTheme.ink,
    required this.icon,
    this.blockColor,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final fill = blockColor ?? (isDark ? c.navy800 : Colors.white);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: fill,
        borderRadius: BorderRadius.circular(AppTheme.radiusSm),
        border: Border.all(color: c.cardBorder, width: AppTheme.borderWidth),
        boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: valueColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: c.cardBorder, width: 1.5),
            ),
            child: Icon(icon, color: valueColor, size: 20),
          ),
          Text(value,
              style: GoogleFonts.spaceGrotesk(
                  fontSize: 34,
                  fontWeight: FontWeight.w700,
                  color: c.textPrimary,
                  height: 1)),
          Text(label,
              style: GoogleFonts.workSans(
                  fontSize: 12,
                  color: c.textSecondary,
                  fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

/// Section header with optional action
class SectionHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? trailing;

  const SectionHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: Theme.of(context).textTheme.titleLarge),
                if (subtitle != null) ...[
                  const SizedBox(height: 2),
                  Text(subtitle!,
                      style: Theme.of(context).textTheme.bodyMedium),
                ],
              ],
            ),
          ),
          if (trailing != null) trailing!,
        ],
      ),
    );
  }
}

/// Skill chip — flat color block, thick outline. Green for matched,
/// coral for missing, neutral outline otherwise.
class SkillChip extends StatelessWidget {
  final String label;
  final SkillChipType type;

  const SkillChip(
      {super.key, required this.label, this.type = SkillChipType.neutral});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final Color bg, text;

    switch (type) {
      case SkillChipType.matched:
        bg = AppTheme.teal200;
        text = AppTheme.tealDark;
        break;
      case SkillChipType.missing:
        bg = AppTheme.red400.withOpacity(0.28);
        text = Theme.of(context).brightness == Brightness.dark
            ? AppTheme.red400
            : const Color(0xFF8C2A1C);
        break;
      case SkillChipType.neutral:
        bg = Theme.of(context).brightness == Brightness.dark
            ? c.navy700
            : Colors.white;
        text = c.textSecondary;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.cardBorder, width: 1.75),
      ),
      child: Text(label,
          style: GoogleFonts.workSans(
              fontSize: 12, color: text, fontWeight: FontWeight.w600)),
    );
  }
}

enum SkillChipType { matched, missing, neutral }

/// Bold divider with a boxed label — like a section tab.
class LabeledDivider extends StatelessWidget {
  final String label;

  const LabeledDivider({super.key, required this.label});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: c.brand,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: c.cardBorder, width: 1.75),
          ),
          child: Text(
            label.toUpperCase(),
            style: GoogleFonts.workSans(
                fontSize: 10,
                color: AppTheme.ink,
                letterSpacing: 1,
                fontWeight: FontWeight.w700),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(child: Container(height: 2, color: c.divider)),
      ],
    );
  }
}

/// Match score stamp — a bold outlined block with the percentage, sitting
/// at a slight brutalist tilt like a stamped sticker.
class AccuracyRing extends StatelessWidget {
  final String accuracy; // e.g. "87.50%"
  final double size;

  const AccuracyRing({super.key, required this.accuracy, this.size = 80});

  double get _value {
    final clean = accuracy.replaceAll('%', '').trim();
    return (double.tryParse(clean) ?? 0) / 100;
  }

  String get _whole => accuracy.replaceAll('%', '').split('.').first;

  Color get _color {
    final v = _value;
    if (v >= 0.75) return AppTheme.teal500;
    if (v >= 0.5) return AppTheme.amber500;
    return AppTheme.red500;
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Transform.rotate(
      angle: -0.08,
      child: Container(
        width: size,
        height: size,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: _color,
          shape: BoxShape.circle,
          border: Border.all(color: c.cardBorder, width: AppTheme.borderWidth),
          boxShadow: AppTheme.hardShadow(c.cardBorder, dx: 4, dy: 4),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '$_whole%',
              style: GoogleFonts.spaceGrotesk(
                  fontSize: size * 0.24,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.ink),
            ),
            Text('matched',
                style: AppTheme.mono(
                    fontSize: size * 0.1,
                    weight: FontWeight.w600,
                    color: AppTheme.ink)),
          ],
        ),
      ),
    );
  }
}

/// User avatar initials block — flat color square, thick outline.
class AvatarCircle extends StatelessWidget {
  final String initials;
  final double size;
  final bool isSelected;

  const AvatarCircle({
    super.key,
    required this.initials,
    this.size = 44,
    this.isSelected = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 150),
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: isSelected ? c.brand : (isDark ? c.navy700 : Colors.white),
        borderRadius: BorderRadius.circular(size * 0.28),
        border: Border.all(
          color: c.cardBorder,
          width: isSelected ? AppTheme.borderWidthThick : AppTheme.borderWidth,
        ),
        boxShadow:
            isSelected ? AppTheme.hardShadow(c.cardBorder, dx: 3, dy: 3) : null,
      ),
      child: Center(
        child: Text(
          initials,
          style: GoogleFonts.spaceGrotesk(
            fontSize: size * 0.34,
            fontWeight: FontWeight.w700,
            color: isSelected ? AppTheme.ink : c.textSecondary,
          ),
        ),
      ),
    );
  }
}

/// Loading placeholder block — flat color sweep instead of a soft shimmer.
class ShimmerCard extends StatefulWidget {
  final double height;

  const ShimmerCard({super.key, this.height = 80});

  @override
  State<ShimmerCard> createState() => _ShimmerCardState();
}

class _ShimmerCardState extends State<ShimmerCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1400))
      ..repeat();
    _anim = Tween<double>(begin: -1, end: 2)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) {
        return Container(
          height: widget.height,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(AppTheme.radiusSm),
            border:
                Border.all(color: c.cardBorder, width: AppTheme.borderWidth),
            gradient: LinearGradient(
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
              stops: [
                (_anim.value - 0.3).clamp(0.0, 1.0),
                _anim.value.clamp(0.0, 1.0),
                (_anim.value + 0.3).clamp(0.0, 1.0),
              ],
              colors: [c.navy800, c.navy600, c.navy800],
            ),
          ),
        );
      },
    );
  }
}

/// Bold outlined badge block
class TealBadge extends StatelessWidget {
  final String text;

  const TealBadge({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(
        color: c.brand,
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: c.cardBorder, width: 1.75),
      ),
      child: Text(
        text,
        style: GoogleFonts.workSans(
            fontSize: 11, color: AppTheme.ink, fontWeight: FontWeight.w700),
      ),
    );
  }
}
