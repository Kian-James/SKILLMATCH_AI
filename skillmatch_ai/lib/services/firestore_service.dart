// lib/services/firestore_service.dart

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';

import '../models/models.dart';

class FirestoreService {
  static const String _collection = 'resume_scans';

  final FirebaseFirestore _db = FirebaseFirestore.instance;

  /// Save a scan record — persists only skills, certs, and identity metadata.
  Future<String> saveScanRecord(ResumeScanRecord record) async {
    final ref = await _db.collection(_collection).add(record.toFirestore());
    return ref.id;
  }

  /// Save a scan record, overwriting any existing record for the same
  /// (userId, fileName, targetJobId) instead of creating a duplicate.
  /// Full analysis always re-runs the pipeline fresh (see upload_screen's
  /// _runFull), so without this, every re-analysis of the same resume piled
  /// up a brand-new history entry instead of refreshing the existing one.
  Future<String> upsertScanRecord(ResumeScanRecord record) async {
    final existing = await _db
        .collection(_collection)
        .where('user_id', isEqualTo: record.userId)
        .where('file_name', isEqualTo: record.fileName)
        .where('target_job_id', isEqualTo: record.targetJobId)
        .limit(1)
        .get();

    if (existing.docs.isNotEmpty) {
      final docId = existing.docs.first.id;
      await _db.collection(_collection).doc(docId).set(record.toFirestore());
      return docId;
    }

    final ref = await _db.collection(_collection).add(record.toFirestore());
    return ref.id;
  }

  /// Check if the same file was already scanned for the same target job.
  /// Returns the stored record (with skills + certs) or null if not found.
  /// The caller can use this to skip the API call and show the progression
  /// chart, but should still hit the API if full gap analysis is needed.
  Future<ResumeScanRecord?> findCachedScan({
    required String userId,
    required String fileName,
    required String targetJobId,
  }) async {
    final query = await _db
        .collection(_collection)
        .where('user_id', isEqualTo: userId)
        .where('file_name', isEqualTo: fileName)
        .where('target_job_id', isEqualTo: targetJobId)
        .orderBy('scanned_at', descending: true)
        .limit(1)
        .get();

    if (query.docs.isEmpty) return null;
    final doc = query.docs.first;
    return ResumeScanRecord.fromFirestore(doc.id, doc.data());
  }

  /// Fetch all scan records for a user, newest first.
  /// Each record contains skills + certs — enough for the progression chart.
  Future<List<ResumeScanRecord>> getUserHistory(String userId) async {
    final query = await _db
        .collection(_collection)
        .where('user_id', isEqualTo: userId)
        .orderBy('scanned_at', descending: true)
        .get();

    return query.docs
        .map((doc) => ResumeScanRecord.fromFirestore(doc.id, doc.data()))
        .toList();
  }

  /// Real-time stream of a user's scan records for the history screen.
  Stream<List<ResumeScanRecord>> watchUserHistory(String userId) {
    return _db
        .collection(_collection)
        .where('user_id', isEqualTo: userId)
        .orderBy('scanned_at', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((doc) => ResumeScanRecord.fromFirestore(doc.id, doc.data()))
            .toList());
  }

  /// Delete a single scan record by Firestore document ID.
  Future<void> deleteScanRecord(String docId) async {
    await _db.collection(_collection).doc(docId).delete();
  }

  /// Update the skills list on an existing scan record.
  Future<void> updateSkills(String docId, List<String> skills) async {
    await _db.collection(_collection).doc(docId).update({'skills': skills});
  }

  /// Update the certs list on an existing scan record.
  Future<void> updateCerts(String docId, List<String> certs) async {
    await _db.collection(_collection).doc(docId).update({'certs': certs});
  }

  /// Canonical job/career titles from the public `jobs` collection —
  /// readable by any authenticated user (no owner check), so this is the
  /// right source for a broad autosuggest list rather than mining
  /// `resume_scans`, which is locked to each user's own docs.
  Future<List<String>> fetchJobsTitles({int limit = 2000}) async {
    final snapshot = await _db.collection('jobs').limit(limit).get();

    debugPrint('[autosuggest] jobs raw doc count: ${snapshot.docs.length}');

    final seen = <String>{};
    final titles = <String>[];
    for (final doc in snapshot.docs) {
      final data = doc.data();
      final raw = data['title'] ?? data['job_title'] ?? data['name'];
      final title = (raw ?? '').toString().trim();
      if (title.isEmpty) continue;
      if (seen.add(title.toLowerCase())) titles.add(title);
    }
    titles.sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
    return titles;
  }

  /// Distinct target-job titles from the *current user's own* scan history —
  /// scoped to [userId] because resume_scans' security rules only allow a
  /// user to read their own docs (a query without this filter is rejected
  /// with permission-denied before it even runs). Useful to merge alongside
  /// [fetchJobsTitles] so a user's own previously-typed titles surface
  /// too, even if they're not in the canonical dataset.
  Future<List<String>> fetchOwnTargetJobTitles({
    required String userId,
    int limit = 500,
  }) async {
    final snapshot = await _db
        .collection(_collection)
        .where('user_id', isEqualTo: userId)
        .limit(limit)
        .get();

    final seen = <String>{};
    final titles = <String>[];
    for (final doc in snapshot.docs) {
      final title = (doc.data()['target_job_title'] as String?)?.trim();
      if (title == null || title.isEmpty) continue;
      if (seen.add(title.toLowerCase())) titles.add(title);
    }
    return titles;
  }

  /// Update the missing-skills list on an existing scan record.
  /// Used when the user manually adds/removes a gap on the results screen —
  /// without this, those edits only ever lived in local widget state.
  Future<void> updateMissingSkills(
      String docId, List<String> missingSkills) async {
    await _db
        .collection(_collection)
        .doc(docId)
        .update({'missing_skills': missingSkills});
  }

  /// Update the matched-skills list on an existing scan record.
  /// Used when the user promotes a missing skill to "I have this" or
  /// removes a matched skill the AI got wrong (a hallucinated match).
  Future<void> updateMatchedSkills(
      String docId, List<String> matchedSkills) async {
    await _db
        .collection(_collection)
        .doc(docId)
        .update({'matched_skills': matchedSkills});
  }
}
