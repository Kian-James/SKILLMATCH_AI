// lib/screens/history_screen.dart
// Shows all past scans for a user with the skill progression chart

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/firestore_service.dart';
import '../theme.dart';
import '../widgets/common_widgets.dart';
import '../widgets/progression_chart.dart';
import 'full_results_screen.dart';
import 'results_screen.dart';

class HistoryScreen extends StatelessWidget {
  final ResumeUser user;

  const HistoryScreen({super.key, required this.user});

  @override
  Widget build(BuildContext context) {
    final _firestore = FirestoreService();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: AvatarCircle(initials: user.avatarInitials, size: 34),
          ),
        ],
      ),
      body: StreamBuilder<List<ResumeScanRecord>>(
        stream: _firestore.watchUserHistory(user.id),
        builder: (context, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return _loadingState();
          }
          if (snap.hasError) {
            return _errorState(context, snap.error.toString());
          }

          final records = snap.data ?? [];

          if (records.isEmpty) {
            return _emptyState(context);
          }

          return CustomScrollView(
            slivers: [
              // ── Skill Progression Chart ──────────────────────────────────
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionHeader(context, 'Skill Progression',
                          '${records.length} scan${records.length != 1 ? 's' : ''} tracked'),
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: AppColors.of(context).navy800,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                              color: AppColors.of(context).cardBorder),
                        ),
                        child: ProgressionChart(history: records),
                      ),
                      const SizedBox(height: 28),
                      _sectionHeader(context, 'All Scans', 'Newest first'),
                      const SizedBox(height: 4),
                    ],
                  ),
                ),
              ),

              // ── Scan record list ─────────────────────────────────────────
              SliverList(
                delegate: SliverChildBuilderDelegate(
                  (context, i) {
                    final r = records[i];
                    return Padding(
                      padding: EdgeInsets.fromLTRB(20, i == 0 ? 8 : 0, 20, 12),
                      child: _ScanCard(
                        record: r,
                        user: user,
                        onDelete: () =>
                            FirestoreService().deleteScanRecord(r.id),
                      ),
                    );
                  },
                  childCount: records.length,
                ),
              ),

              const SliverToBoxAdapter(child: SizedBox(height: 32)),
            ],
          );
        },
      ),
    );
  }

  Widget _sectionHeader(BuildContext ctx, String title, String sub) =>
      SectionHeader(title: title, subtitle: sub);

  Widget _loadingState() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(children: [
        const SizedBox(height: 20),
        const ShimmerCard(height: 220),
        const SizedBox(height: 16),
        const ShimmerCard(height: 90),
        const SizedBox(height: 12),
        const ShimmerCard(height: 90),
      ]),
    );
  }

  Widget _errorState(BuildContext context, String err) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.cloud_off_rounded,
              color: AppColors.of(context).textTertiary, size: 48),
          const SizedBox(height: 16),
          Text('Failed to load history',
              style: GoogleFonts.spaceGrotesk(
                  color: AppColors.of(context).textSecondary,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(err,
              style: GoogleFonts.workSans(
                  color: AppColors.of(context).textTertiary, fontSize: 12),
              textAlign: TextAlign.center),
        ]),
      ),
    );
  }

  Widget _emptyState(BuildContext ctx) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: AppColors.of(ctx).navy800,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: AppColors.of(ctx).cardBorder, width: 2.5),
            ),
            child: Icon(Icons.history_rounded,
                color: AppColors.of(ctx).textTertiary, size: 36),
          ),
          const SizedBox(height: 20),
          Text('No scans yet', style: Theme.of(ctx).textTheme.headlineMedium),
          const SizedBox(height: 8),
          Text(
              'Upload your first resume to start tracking\nyour skill growth over time.',
              style: Theme.of(ctx).textTheme.bodyMedium,
              textAlign: TextAlign.center),
        ]),
      ),
    );
  }
}

// ── Individual scan record card ──────────────────────────────────────────────

class _ScanCard extends StatelessWidget {
  final ResumeScanRecord record;
  final ResumeUser user;
  final VoidCallback onDelete;

  const _ScanCard({
    required this.record,
    required this.user,
    required this.onDelete,
  });

  double get _accuracy {
    final c = record.nlpAccuracy.replaceAll('%', '').trim();
    return double.tryParse(c) ?? 0;
  }

  Color get _accuracyColor {
    if (_accuracy >= 75) return AppTheme.green500;
    if (_accuracy >= 50) return AppTheme.amber500;
    return AppTheme.red500;
  }

  String get _modeIcon => switch (record.uploadMode) {
        'pdf' => '📄',
        'image' => '🖼',
        'scan' => '📷',
        _ => '📄',
      };

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => record.analysisType == AnalysisType.full
              ? FullResultsScreen(record: record, user: user, fromCache: true)
              : ResultsScreen(record: record, user: user, fromCache: true),
        ),
      ),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.of(context).navy800,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.of(context).cardBorder, width: 2.5),
          boxShadow: AppTheme.hardShadow(AppColors.of(context).cardBorder, dx: 4, dy: 4),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          // Top row
          Row(children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: AppTheme.teal500.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Center(
                child: Text(_modeIcon, style: const TextStyle(fontSize: 18)),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(record.displayTitle,
                      style: Theme.of(context).textTheme.titleMedium,
                      overflow: TextOverflow.ellipsis),
                  Text(record.fileName,
                      style: GoogleFonts.workSans(
                          fontSize: 11,
                          color: AppColors.of(context).textTertiary,
                          fontStyle: FontStyle.italic),
                      overflow: TextOverflow.ellipsis),
                ])),
            // Delete
            GestureDetector(
              onTap: () => _confirmDelete(context),
              child: Icon(Icons.delete_outline_rounded,
                  color: AppColors.of(context).textTertiary, size: 18),
            ),
          ]),

          const SizedBox(height: 14),

          // Stats row
          Row(children: [
            Expanded(
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(children: [
                  _pill('${record.skillCount}', 'skills', AppTheme.teal500),
                  const SizedBox(width: 8),
                  _pill('${record.matchedCount}', 'matched', AppTheme.green500),
                  const SizedBox(width: 8),
                  _pill('${record.missingCount}', 'missing', AppTheme.red500),
                ]),
              ),
            ),
            const SizedBox(width: 8),
            // Accuracy badge — always fully visible, never pushed off-screen
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: _accuracyColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: _accuracyColor.withOpacity(0.3), width: 2.5),
              ),
              child: Text(record.nlpAccuracy,
                  style: GoogleFonts.spaceGrotesk(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: _accuracyColor)),
            ),
          ]),

          const SizedBox(height: 10),
          Divider(height: 1, color: AppColors.of(context).divider),
          const SizedBox(height: 8),

          // Date + industry
          Row(children: [
            Icon(Icons.schedule_rounded,
                size: 12, color: AppColors.of(context).textTertiary),
            const SizedBox(width: 5),
            Text(
              DateFormat('MMM d, yyyy  hh:mm a').format(record.scannedAt),
              style: GoogleFonts.workSans(
                  fontSize: 11, color: AppColors.of(context).textTertiary),
            ),
            if (record.industry.isNotEmpty) ...[
              const SizedBox(width: 10),
              Text('•',
                  style: TextStyle(
                      color: AppColors.of(context).textTertiary, fontSize: 11)),
              const SizedBox(width: 10),
              Flexible(
                child: Text(record.industry,
                    style: GoogleFonts.workSans(
                        fontSize: 11,
                        color: AppColors.of(context).textTertiary),
                    overflow: TextOverflow.ellipsis),
              ),
            ],
          ]),
        ]),
      ),
    );
  }

  Widget _pill(String value, String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Text(value,
            style: GoogleFonts.spaceGrotesk(
                fontSize: 12, fontWeight: FontWeight.w700, color: color)),
        const SizedBox(width: 4),
        Text(label,
            style: GoogleFonts.workSans(
                fontSize: 11, color: color.withOpacity(0.75))),
      ]),
    );
  }

  void _confirmDelete(BuildContext context) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.of(context).navy700,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Delete Scan?',
            style: GoogleFonts.spaceGrotesk(
                color: AppColors.of(context).textPrimary,
                fontWeight: FontWeight.w600)),
        content: Text(
          'This will permanently remove the scan for "${record.displayTitle}".',
          style: GoogleFonts.workSans(
              color: AppColors.of(context).textSecondary, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel',
                style: GoogleFonts.workSans(
                    color: AppColors.of(context).textSecondary)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.red500,
                foregroundColor: Colors.white),
            onPressed: () {
              Navigator.pop(context);
              onDelete();
            },
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}
