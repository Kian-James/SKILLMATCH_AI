// lib/models/models.dart

// ── Analysis type ─────────────────────────────────────────────────────────────
enum AnalysisType { targeted, full }

// ── User ─────────────────────────────────────────────────────────────────────
class ResumeUser {
  final String id; // Firebase UID (or 'guest_<timestamp>' for guests)
  final String name;
  final String email;
  final String avatarInitials;
  final String? photoUrl;
  final bool isGuest;

  const ResumeUser({
    required this.id,
    required this.name,
    required this.email,
    required this.avatarInitials,
    this.photoUrl,
    this.isGuest = false,
  });

  /// Build from a Firebase Auth user
  factory ResumeUser.fromFirebase({
    required String uid,
    required String? displayName,
    required String? email,
    required String? photoUrl,
  }) {
    final name = displayName ?? email?.split('@').first ?? 'User';
    final parts = name.trim().split(' ');
    final initials = parts.length >= 2
        ? '${parts[0][0]}${parts[1][0]}'.toUpperCase()
        : parts[0].substring(0, parts[0].length.clamp(0, 2)).toUpperCase();
    return ResumeUser(
      id: uid,
      name: name,
      email: email ?? '',
      avatarInitials: initials,
      photoUrl: photoUrl,
      isGuest: false,
    );
  }

  /// Guest user — scoped by a timestamp so each session is isolated
  factory ResumeUser.guest() {
    return ResumeUser(
      id: 'guest_${DateTime.now().millisecondsSinceEpoch}',
      name: 'Guest',
      email: '',
      avatarInitials: 'G',
      isGuest: true,
    );
  }

  /// Legacy helper used in history/Firestore queries
  factory ResumeUser.fromName(String name) {
    final parts = name.trim().split(' ');
    final initials = parts.length >= 2
        ? '${parts[0][0]}${parts[1][0]}'.toUpperCase()
        : parts[0].substring(0, parts[0].length.clamp(0, 2)).toUpperCase();
    return ResumeUser(
      id: name.toLowerCase().replaceAll(' ', '_'),
      name: name,
      email: '',
      avatarInitials: initials,
    );
  }
}

// ── Shared sub-models ─────────────────────────────────────────────────────────

class SkillGapAnalysis {
  final List<String> requiredSkills;
  final List<String> matchedSkills;
  final List<String> missingSkills;
  final String nlpAccuracy;
  final double matchRatio;
  final double hybridScore;

  const SkillGapAnalysis({
    required this.requiredSkills,
    required this.matchedSkills,
    required this.missingSkills,
    required this.nlpAccuracy,
    required this.matchRatio,
    required this.hybridScore,
  });

  factory SkillGapAnalysis.fromJson(Map<String, dynamic> j) => SkillGapAnalysis(
        requiredSkills: List<String>.from(j['required_skills'] ?? []),
        matchedSkills: List<String>.from(j['matched_skills'] ?? []),
        missingSkills: List<String>.from(j['missing_skills'] ?? []),
        nlpAccuracy: j['nlp_accuracy'] ?? '0%',
        matchRatio: (j['match_ratio'] ?? 0.0).toDouble(),
        hybridScore: (j['hybrid_score'] ?? 0.0).toDouble(),
      );
}

class CertGapAnalysis {
  final List<String> requiredCerts;
  final List<String> matchedCerts;
  final List<String> missingCerts;

  const CertGapAnalysis(
      {required this.requiredCerts,
      required this.matchedCerts,
      required this.missingCerts});

  factory CertGapAnalysis.fromJson(Map<String, dynamic> j) => CertGapAnalysis(
        requiredCerts: List<String>.from(j['required_certs'] ?? []),
        matchedCerts: List<String>.from(j['matched_certs'] ?? []),
        missingCerts: List<String>.from(j['missing_certs'] ?? []),
      );
}

class TargetJob {
  final String jobId;
  final String title;
  final String industry;

  const TargetJob(
      {required this.jobId, required this.title, required this.industry});

  factory TargetJob.fromJson(Map<String, dynamic> j) => TargetJob(
      jobId: j['job_id'] ?? '',
      title: j['title'] ?? '',
      industry: j['industry'] ?? '');
}

class ResumeSkillsMeta {
  final List<String> discoveredSkills;
  final List<String> discoveredCerts;
  final int totalSkillsFound;
  final int totalCertsFound;

  const ResumeSkillsMeta({
    required this.discoveredSkills,
    required this.discoveredCerts,
    required this.totalSkillsFound,
    required this.totalCertsFound,
  });

  factory ResumeSkillsMeta.fromJson(Map<String, dynamic> j) => ResumeSkillsMeta(
        discoveredSkills: List<String>.from(j['discovered_skills'] ?? []),
        discoveredCerts: List<String>.from(j['discovered_certs'] ?? []),
        totalSkillsFound: j['total_skills_found'] ?? 0,
        totalCertsFound: j['total_certs_found'] ?? 0,
      );
}

class TextAcquisition {
  final String textSource;
  final String ocrQuality;
  final int charCount;
  final String rawText;

  const TextAcquisition({
    required this.textSource,
    required this.ocrQuality,
    required this.charCount,
    required this.rawText,
  });

  factory TextAcquisition.fromJson(Map<String, dynamic> j) => TextAcquisition(
        textSource: j['text_source'] ?? '',
        ocrQuality: j['ocr_quality'] ?? '0%',
        charCount: j['char_count'] ?? 0,
        rawText: j['raw_text'] ?? '',
      );
}

// ── /target-career-analysis response ─────────────────────────────────────────

class TargetCareerResult {
  final TargetJob targetJob;
  final ResumeSkillsMeta resumeSkills;
  final SkillGapAnalysis skillGap;
  final CertGapAnalysis certGap;
  final TextAcquisition textAcquisition;

  const TargetCareerResult({
    required this.targetJob,
    required this.resumeSkills,
    required this.skillGap,
    required this.certGap,
    required this.textAcquisition,
  });

  factory TargetCareerResult.fromJson(Map<String, dynamic> j) =>
      TargetCareerResult(
        targetJob: TargetJob.fromJson(j['target_job'] ?? {}),
        resumeSkills: ResumeSkillsMeta.fromJson(j['resume_skills'] ?? {}),
        skillGap: SkillGapAnalysis.fromJson(j['skill_gap_analysis'] ?? {}),
        certGap: CertGapAnalysis.fromJson(j['cert_gap_analysis'] ?? {}),
        textAcquisition: TextAcquisition.fromJson(j['text_acquisition'] ?? {}),
      );
}

// ── /thesis-full-analysis response ───────────────────────────────────────────

/// One entry in the Top 5 NLP career recommendations
class Top5Match {
  final int rank;
  final String jobId;
  final String title;
  final String industry;
  final String nlpAccuracy;
  final List<String> matchedSkills;
  final List<String> unmatchedSkills;
  final int groundTruthCount;
  final String fitReason;

  const Top5Match({
    required this.rank,
    required this.jobId,
    required this.title,
    required this.nlpAccuracy,
    required this.matchedSkills,
    required this.unmatchedSkills,
    required this.groundTruthCount,
    this.industry = '',
    this.fitReason = '',
  });

  double get accuracyValue {
    final c = nlpAccuracy.replaceAll('%', '').trim();
    return double.tryParse(c) ?? 0;
  }

  factory Top5Match.fromJson(Map<String, dynamic> j) => Top5Match(
        rank: j['rank'] ?? 0,
        jobId: j['job_id'] ?? '',
        title: j['title'] ?? '',
        industry: j['industry'] ?? '',
        nlpAccuracy: j['nlp_accuracy'] ?? '0%',
        matchedSkills: List<String>.from(j['matched_skills'] ?? []),
        unmatchedSkills: List<String>.from(j['unmatched_skills'] ?? []),
        groundTruthCount: j['ground_truth_count'] ?? 0,
        fitReason: j['fit_reason'] ?? '',
      );
}

class FullAnalysisResult {
  // Stage 1
  final String textSource;
  final String ocrQuality;
  final int charCount;
  final String rawText;

  // Stage 2 — skills
  final String llmModel;
  final List<String> discoveredSkills;

  // Stage 2B — certs
  final List<String> discoveredCerts;

  // Stage 3 — NLP
  final String nlpAccuracy; // best job accuracy
  final List<String> matchedSkills;
  final List<String> unmatchedSkills;
  final List<Top5Match> top5;

  // Stage 4 — FAISS
  final String faissScore;
  final int faissIndexTotal;

  const FullAnalysisResult({
    required this.textSource,
    required this.ocrQuality,
    required this.charCount,
    required this.rawText,
    required this.llmModel,
    required this.discoveredSkills,
    required this.discoveredCerts,
    required this.nlpAccuracy,
    required this.matchedSkills,
    required this.unmatchedSkills,
    required this.top5,
    required this.faissScore,
    required this.faissIndexTotal,
  });

  factory FullAnalysisResult.fromJson(Map<String, dynamic> j) {
    final s1 = j['STAGE_1_TEXT_ACQUISITION'] as Map<String, dynamic>? ?? {};
    final s2 = j['STAGE_2_SKILL_EXTRACTION'] as Map<String, dynamic>? ?? {};
    final s2b = j['STAGE_2B_CERT_EXTRACTION'] as Map<String, dynamic>? ?? {};
    final s3 = j['STAGE_3_NLP_VALIDATION'] as Map<String, dynamic>? ?? {};
    final s4 = j['STAGE_4_FAISS_INTEGRITY'] as Map<String, dynamic>? ?? {};

    final top5Raw =
        s3['Top_5_Career_Recommendations_NLP'] as List<dynamic>? ?? [];

    return FullAnalysisResult(
      textSource: s1['text_source'] ?? '',
      ocrQuality: s1['ocr_quality'] ?? '0%',
      charCount: s1['char_count'] ?? 0,
      rawText: s1['raw_text'] ?? '',
      llmModel: s2['llm_model'] ?? '',
      discoveredSkills: List<String>.from(s2['discovered_skills'] ?? []),
      discoveredCerts: List<String>.from(s2b['discovered_certificates'] ?? []),
      nlpAccuracy: s3['nlp_accuracy'] ?? '0%',
      matchedSkills: List<String>.from(s3['matched_skills'] ?? []),
      unmatchedSkills: List<String>.from(s3['unmatched_skills'] ?? []),
      top5: top5Raw
          .map((e) => Top5Match.fromJson(e as Map<String, dynamic>))
          .toList(),
      faissScore: s4['score'] ?? '0%',
      faissIndexTotal: s4['index_total'] ?? 0,
    );
  }
}

// ── Firestore scan record ─────────────────────────────────────────────────────

class ResumeScanRecord {
  final String id;
  final String userId;
  final String fileName;

  // analysisType determines which fields are populated
  final AnalysisType analysisType;

  // Targeted fields
  final String targetJobTitle;
  final String targetJobId;
  final String industry;
  final List<String> matchedSkills;
  final List<String> missingSkills;
  final String nlpAccuracy;
  final double hybridScore;

  // Full analysis fields
  final List<Top5Match> top5Matches;
  final String faissScore;

  // Shared
  final List<String> skills;
  final List<String> certs;
  final DateTime scannedAt;
  final String uploadMode; // 'pdf' | 'image' | 'scan'

  const ResumeScanRecord({
    required this.id,
    required this.userId,
    required this.fileName,
    required this.analysisType,
    required this.skills,
    required this.certs,
    required this.scannedAt,
    this.uploadMode = 'pdf',
    // targeted
    this.targetJobTitle = '',
    this.targetJobId = '',
    this.industry = '',
    this.matchedSkills = const [],
    this.missingSkills = const [],
    this.nlpAccuracy = '0%',
    this.hybridScore = 0.0,
    // full
    this.top5Matches = const [],
    this.faissScore = '0%',
  });

  int get skillCount => skills.length;
  int get certCount => certs.length;
  int get matchedCount => matchedSkills.length;
  int get missingCount => missingSkills.length;

  String get displayTitle =>
      analysisType == AnalysisType.targeted ? targetJobTitle : 'Full Analysis';

  ResumeScanRecord copyWith({
    String? id,
    String? userId,
    String? fileName,
    AnalysisType? analysisType,
    List<String>? skills,
    List<String>? certs,
    DateTime? scannedAt,
    String? uploadMode,
    String? targetJobTitle,
    String? targetJobId,
    String? industry,
    List<String>? matchedSkills,
    List<String>? missingSkills,
    String? nlpAccuracy,
    double? hybridScore,
    List<Top5Match>? top5Matches,
    String? faissScore,
  }) =>
      ResumeScanRecord(
        id: id ?? this.id,
        userId: userId ?? this.userId,
        fileName: fileName ?? this.fileName,
        analysisType: analysisType ?? this.analysisType,
        skills: skills ?? this.skills,
        certs: certs ?? this.certs,
        scannedAt: scannedAt ?? this.scannedAt,
        uploadMode: uploadMode ?? this.uploadMode,
        targetJobTitle: targetJobTitle ?? this.targetJobTitle,
        targetJobId: targetJobId ?? this.targetJobId,
        industry: industry ?? this.industry,
        matchedSkills: matchedSkills ?? this.matchedSkills,
        missingSkills: missingSkills ?? this.missingSkills,
        nlpAccuracy: nlpAccuracy ?? this.nlpAccuracy,
        hybridScore: hybridScore ?? this.hybridScore,
        top5Matches: top5Matches ?? this.top5Matches,
        faissScore: faissScore ?? this.faissScore,
      );

  /// Saves only discovered skills and certifications to Firestore.
  /// Everything else (matched, missing, scores, top5) lives in the
  /// API response for the current session only and is never persisted.
  Map<String, dynamic> toFirestore() => {
        'user_id': userId,
        'file_name': fileName,
        'scanned_at': scannedAt.toIso8601String(),
        'upload_mode': uploadMode,
        'analysis_type': analysisType.name,
        'target_job_title': targetJobTitle,
        'target_job_id': targetJobId,
        'industry': industry,
        // ── Always stored ──────────────────────────────────────────────────────
        'skills': skills,
        'certs': certs,
        // ── Targeted analysis ──────────────────────────────────────────────────
        'matched_skills': matchedSkills,
        'missing_skills': missingSkills,
        'nlp_accuracy': nlpAccuracy,
        'hybrid_score': hybridScore,
        // ── Full analysis ──────────────────────────────────────────────────────
        'faiss_score': faissScore,
        'top5_matches': top5Matches
            .map((m) => {
                  'rank': m.rank,
                  'job_id': m.jobId,
                  'title': m.title,
                  'industry': m.industry,
                  'nlp_accuracy': m.nlpAccuracy,
                  'matched_skills': m.matchedSkills,
                  'unmatched_skills': m.unmatchedSkills,
                  'ground_truth_count': m.groundTruthCount,
                  'fit_reason': m.fitReason,
                })
            .toList(),
      };

  factory ResumeScanRecord.fromFirestore(String id, Map<String, dynamic> d) {
    final aType = d['analysis_type'] == 'full'
        ? AnalysisType.full
        : AnalysisType.targeted;

    final top5Raw = d['top5_matches'] as List<dynamic>? ?? [];

    return ResumeScanRecord(
      id: id,
      userId: d['user_id'] ?? '',
      fileName: d['file_name'] ?? '',
      analysisType: aType,
      scannedAt: d['scanned_at'] != null
          ? DateTime.parse(d['scanned_at'])
          : DateTime.now(),
      uploadMode: d['upload_mode'] ?? 'pdf',
      targetJobTitle: d['target_job_title'] ?? '',
      targetJobId: d['target_job_id'] ?? '',
      industry: d['industry'] ?? '',
      skills: List<String>.from(d['skills'] ?? []),
      certs: List<String>.from(d['certs'] ?? []),
      matchedSkills: List<String>.from(d['matched_skills'] ?? []),
      missingSkills: List<String>.from(d['missing_skills'] ?? []),
      nlpAccuracy: d['nlp_accuracy'] ?? '0%',
      hybridScore: (d['hybrid_score'] ?? 0.0).toDouble(),
      faissScore: d['faiss_score'] ?? '0%',
      top5Matches: top5Raw
          .map((e) => Top5Match.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}
