// lib/services/firestore_service.dart

import 'package:cloud_firestore/cloud_firestore.dart';

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
}
