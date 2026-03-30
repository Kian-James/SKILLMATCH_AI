// widgets/skill_chip.dart
// Reusable skill chip with three visual states: present, missing, certification.
// Missing chips are tappable to trigger learning resource lookup.

import 'package:flutter/material.dart';

enum SkillChipType { present, missing, cert }

class SkillChip extends StatelessWidget {
  final String skill;
  final SkillChipType type;
  final VoidCallback? onTap;

  const SkillChip({
    super.key,
    required this.skill,
    required this.type,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    final (Color bg, Color fg, IconData icon) = switch (type) {
      SkillChipType.present => (
          cs.secondaryContainer,
          cs.onSecondaryContainer,
          Icons.check_circle_outline_rounded,
        ),
      SkillChipType.missing => (
          cs.errorContainer,
          cs.onErrorContainer,
          Icons.add_circle_outline_rounded,
        ),
      SkillChipType.cert => (
          cs.tertiaryContainer,
          cs.onTertiaryContainer,
          Icons.verified_outlined,
        ),
    };

    final chip = Chip(
      avatar: Icon(icon, size: 15, color: fg),
      label: Text(
        skill,
        style: TextStyle(
          color: fg,
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
      backgroundColor: bg,
      side: BorderSide.none,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
      visualDensity: VisualDensity.compact,
      padding: const EdgeInsets.symmetric(horizontal: 4),
    );

    if (onTap != null) {
      return GestureDetector(
        onTap: onTap,
        child: Tooltip(
          message: 'Tap to see learning resources',
          child: chip,
        ),
      );
    }

    return chip;
  }
}
