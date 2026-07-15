// lib/widgets/common_widgets.dart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';

/// Glowing stat metric card
class StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color valueColor;
  final IconData icon;

  const StatCard({
    super.key,
    required this.label,
    required this.value,
    this.valueColor = AppTheme.teal500,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.navy800,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: c.textTertiary, size: 18),
          const SizedBox(height: 10),
          Text(value,
              style: GoogleFonts.fraunces(
                  fontSize: 26,
                  fontWeight: FontWeight.w700,
                  color: valueColor,
                  height: 1)),
          const SizedBox(height: 4),
          Text(label,
              style: GoogleFonts.workSans(
                  fontSize: 12,
                  color: c.textTertiary,
                  fontWeight: FontWeight.w500)),
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

/// Skill chip — green for matched, red for missing, neutral for default
class SkillChip extends StatelessWidget {
  final String label;
  final SkillChipType type;

  const SkillChip(
      {super.key, required this.label, this.type = SkillChipType.neutral});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    final Color bg, border, text;

    switch (type) {
      case SkillChipType.matched:
        bg = AppTheme.green500.withOpacity(0.12);
        border = AppTheme.green500.withOpacity(0.35);
        text = AppTheme.green400;
        break;
      case SkillChipType.missing:
        bg = AppTheme.red500.withOpacity(0.12);
        border = AppTheme.red500.withOpacity(0.35);
        text = AppTheme.red400;
        break;
      case SkillChipType.neutral:
        bg = c.navy700;
        border = c.cardBorder;
        text = c.textSecondary;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: border, width: 0.8),
      ),
      child: Text(label,
          style: GoogleFonts.workSans(
              fontSize: 12, color: text, fontWeight: FontWeight.w500)),
    );
  }
}

enum SkillChipType { matched, missing, neutral }

/// Teal-accented divider with label
class LabeledDivider extends StatelessWidget {
  final String label;

  const LabeledDivider({super.key, required this.label});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Row(
      children: [
        Expanded(child: Container(height: 1, color: c.divider)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Text(
            label.toUpperCase(),
            style: GoogleFonts.workSans(
                fontSize: 10,
                color: c.textTertiary,
                letterSpacing: 1.2,
                fontWeight: FontWeight.w600),
          ),
        ),
        Expanded(child: Container(height: 1, color: c.divider)),
      ],
    );
  }
}

/// Ink-stamp match seal — the "reviewed" case-file signature.
/// A dashed, slightly rotated ring stamped with the match percentage,
/// echoing a document reviewer's rubber stamp.
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
    if (v >= 0.75) return AppTheme.green500;
    if (v >= 0.5) return AppTheme.amber500;
    return AppTheme.red500;
  }

  @override
  Widget build(BuildContext context) {
    return Transform.rotate(
      angle: -0.14, // ~-8°
      child: SizedBox(
        width: size,
        height: size,
        child: CustomPaint(
          painter: _DashedCirclePainter(color: _color),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '$_whole%',
                  style: GoogleFonts.fraunces(
                      fontSize: size * 0.22,
                      fontWeight: FontWeight.w600,
                      color: _color),
                ),
                Text('reviewed',
                    style: AppTheme.mono(
                        fontSize: size * 0.09,
                        weight: FontWeight.w500,
                        color: _color)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _DashedCirclePainter extends CustomPainter {
  final Color color;
  const _DashedCirclePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final radius = size.width / 2 - 1.5;
    final center = Offset(size.width / 2, size.height / 2);
    const dashCount = 28;
    const gapFraction = 0.45;
    for (int i = 0; i < dashCount; i++) {
      final start = (i / dashCount) * 2 * 3.14159265;
      final sweep = (2 * 3.14159265 / dashCount) * (1 - gapFraction);
      canvas.drawArc(Rect.fromCircle(center: center, radius: radius), start,
          sweep, false, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _DashedCirclePainter oldDelegate) =>
      oldDelegate.color != color;
}

/// User avatar initials circle
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
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: LinearGradient(
          colors: isSelected
              ? [AppTheme.teal500, AppTheme.teal300]
              : [c.navy600, c.navy500],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: isSelected
            ? Border.all(color: c.brand, width: 2)
            : Border.all(color: c.cardBorder),
      ),
      child: Center(
        child: Text(
          initials,
          style: GoogleFonts.fraunces(
            fontSize: size * 0.35,
            fontWeight: FontWeight.w700,
            color: isSelected ? AppTheme.navy900 : c.textSecondary,
          ),
        ),
      ),
    );
  }
}

/// Loading shimmer card placeholder
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
            borderRadius: BorderRadius.circular(14),
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

/// Teal pill badge
class TealBadge extends StatelessWidget {
  final String text;

  const TealBadge({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: c.brand.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: c.brand.withOpacity(0.3)),
      ),
      child: Text(
        text,
        style: GoogleFonts.workSans(
            fontSize: 11, color: c.brandSoft, fontWeight: FontWeight.w600),
      ),
    );
  }
}
