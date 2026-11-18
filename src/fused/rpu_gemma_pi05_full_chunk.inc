// Included in GemmaModel's private section. Default-off, exact P544/P800
// experiment: one QKV and one compute callback, without doubling the old
// four-slab allocation. The C272/C400 collective ABIs remain internal tiles.
enum class Pi05FullSite : int64_t {
    Policy = 0, InputCompact, InputNorm, QQuant, QLinear, QRope,
    KvLinear, KvCache, Mask, Attention, OLinear, OwnerNorm, GateUp,
    DownQuant, DownLinear, Residual, Tail, Output
};
static constexpr int64_t PI05_FULL_SITE_BASE = 0x5049303546550000LL;

int64_t pi05_full_b() const { return pi05_pair_rows() * 4096; }
int64_t pi05_full_d() const { return pi05_pair_rows() * 512; }
int64_t pi05_full_gate_scratch() const {
    return 2 * (pi05_prefill_a8_panel512_ ? 512 : pi05_pair_rows() < 400 ? 256 : 384) * 2048;
}
int64_t pi05_full_root_bytes() const {
    return std::max(2 * pi05_full_b(), pi05_a8_root_bytes() + pi05_full_gate_scratch());
}
int64_t pi05_full_mask_bytes() const { return pi05_prefix_rows() * pi05_prefix_rows() * 2; }
int64_t pi05_full_tmp_bytes() const {
    return Align(sdpa_compute_tmp_v16_size(make_sdpa_config(4), pi05_prefix_rows()) * 32, int64_t(256));
}
KernelId pi05_full_norm_kernel() const {
    const auto route = rmsnorm_spm_contract_.resolve(pi05_prefix_rows(), 2048);
    switch (route) {
        case RpuRmsNormSpmRoute::V16:
        case RpuRmsNormSpmRoute::V16_GRID8: return KernelId::RMS_NORM_SPM_V16;
        case RpuRmsNormSpmRoute::V32:
        case RpuRmsNormSpmRoute::V32_GRID8: return KernelId::RMS_NORM_SPM_V32;
        case RpuRmsNormSpmRoute::BASE: return KernelId::RMS_NORM_BF16_SPM;
        default: TORCH_CHECK(false, "Pi full chunk requires the ordinary cold RMSNorm contract");
    }
}
bool pi05_full_chunk_owner() const {
    return pi05_full_chunk_ && !linear_acc32_ && eps_ == 1e-6 &&
        (pi05_pair_rows() == 272 || pi05_pair_rows() == 400) &&
        pi05_prefill_a8_ && pi05_prefill_kv_only_ && pi05_down_a8_enabled() &&
        pi05_prefill_gateup_pair_owner() &&
        pi05_prefill_dma_geometry() && num_cores() == 8 && mlp_tp() == 8;
}
bool pi05_full_chunk_domain(const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout, int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    const auto single = [this](const ResolvedFmbStageChunks& stage) {
        return stage.plan.chunk_size == pi05_prefix_rows() && stage.plan.num_chunks == 1 &&
            stage.chunks.size() == 1 && stage.chunks[0].idx == 0 &&
            stage.chunks[0].offset == 0 && stage.chunks[0].len == pi05_prefix_rows() &&
            stage.chunks[0].kv_seq_len == pi05_prefix_rows();
    };
    return pi05_full_chunk_owner() && physical_len == pi05_prefix_rows() &&
        logical_len == physical_len && position == 0 &&
        layout.chunk_size == physical_len && layout.effective_kv_cs() == physical_len &&
        layout.batch_size == 1 && layout.use_attn_mask && !layout.is_causal &&
        layout.max_kv_seq_len == physical_len &&
        layout.attention_policy == AttentionExecutionPolicy::DDR_KV &&
        chunk_envelope().declared() && chunk_envelope().max_kv_len == 2048 &&
        plan.chunk_mode == ChunkMode::KV_FIRST && single(plan.qkv) && single(plan.compute) &&
        sdpa_call_valid(make_sdpa_config(4), physical_len, 0);
}
std::vector<int64_t> pi05_full_chunk_arguments() const {
    const int64_t c = pi05_pair_rows(), p = pi05_prefix_rows();
    const auto q = rpu_pl_w8a8::installed_candidate(rpu_pl_w8a8::select_tile(
        rpu_pl_w8a8::physical_shape(p,2048,2048,rpu_pl_w8a8::Partition::Col,8,false)),2048);
    const auto down = rpu_pl_w8a8::installed_candidate(rpu_pl_w8a8::select_tile(
        rpu_pl_w8a8::physical_shape(p,2048,16384,rpu_pl_w8a8::Partition::Row,8,false)),2048);
    TORCH_CHECK(q.ordinal == 19 && down.ordinal == 27,
                "Pi full chunk cold/runtime Q19 or Down27 selection drift");
    const auto o = rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear,
        rpu_pl_tiling::WeightType::Int8,false,c,2048,256);
    const auto kv = rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner,
        rpu_pl_tiling::WeightType::Int8,false,c,32,2048);
    const auto gu = rpu_pi_wall::select(pi05_prefill_a8_panel512_
        ? rpu_pi_wall::Operation::A8PairedGeluPanel512 : rpu_pi_wall::Operation::A8PairedGelu,
        rpu_pl_tiling::WeightType::Int8,true,c,2048,2048);
    const auto sdpa = sdpa_compute_tiling(make_sdpa_config(4),p);
    const auto id = [](KernelId value) { return static_cast<int64_t>(value); };
    return {1,p,c,18,17,2048,16384,256,8,8,1,1,
        pi05_full_root_bytes(),2*pi05_full_d(),pi05_full_b(),pi05_full_d(),
        pi05_a8_root_bytes(),pi05_a8_bytes(),pi05_full_gate_scratch(),
        pi05_full_mask_bytes(),pi05_full_tmp_bytes(),
        id(pi05_full_norm_kernel()),id(KernelId::DYNAMIC_QUANTIZE_W8A8),
        id(KernelId::PARALLEL_LINEAR_W8A8_ACC32_RESIDENT_M144N48K512_LK2048_SW_SXFUSED_HOIST),
        19,q.tile.m,q.tile.n,q.tile.k,id(KernelId::PARTIAL_MROPE),
        id(pi05_prefill_kv1_pair_kernel()),kv.tile.m_tile,kv.tile.n_tile,kv.tile.k_tile,kv.grid_x,kv.grid_y,
        id(c==272?KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M272D256P272:KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M400D256P400),
        id(KernelId::SDPA_FLASH_ATTN_SPM),sdpa.tile_m,sdpa.tile_n,sdpa.tile_k,CeilDiv(p,sdpa.tile_m),
        id(pi_wall_kernel_id(o.ordinal)),o.tile.m_tile,o.tile.n_tile,o.tile.k_tile,o.grid_x,o.grid_y,
        id(c==272?KernelId::PI05_OWNER_NORM_COMPACT_A8_M272N2048:KernelId::PI05_OWNER_NORM_COMPACT_A8_M400N2048),
        id(pi05_prefill_gateup_pair_kernel()),gu.tile.m_tile,gu.tile.n_tile,gu.tile.k_tile,gu.grid_x,gu.grid_y,
        id(KernelId::PARALLEL_LINEAR_W8A8_ACC32_RESIDENT_M64N80K256_LK2048_SW_SXFUSED_HOIST),
        27,down.tile.m,down.tile.n,down.tile.k,
        id(c==272?KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M272N2048:KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M400N2048),
        pi05_prefill_a8_panel512_?1:0,0,1,2,3};
}

// This is the single source for both cold descriptors and runtime consumers.
// Root numbers 0/1/2/3 denote residual1/full_r1/full_rc0/full_rc1.
FmbRouteManifestEntry pi05_full_route(Pi05FullSite site, int half=0) const {
    const int64_t c=pi05_pair_rows(),p=pi05_prefix_rows(),b=pi05_full_b(),d=pi05_full_d();
    FmbRouteManifestEntry r{PI05_FULL_SITE_BASE+static_cast<int64_t>(site),
        FmbRouteFamily::GRAPH_SCHEDULE,1,0,{},half};
    switch(site) {
        case Pi05FullSite::Policy: r.arguments=pi05_full_chunk_arguments();break;
        case Pi05FullSite::InputCompact:
            r.family=FmbRouteFamily::COLLECTIVE;r.arguments={0,half*b,2,half*d,d/2,d,8};break;
        case Pi05FullSite::InputNorm:
            r.family=FmbRouteFamily::NORMALIZATION;
            r.selector=static_cast<int64_t>(rmsnorm_spm_contract_.resolve(p,2048));
            r.arguments=rmsnorm_spm_contract_.manifest_arguments(p,2048);break;
        case Pi05FullSite::QQuant:
            r.family=FmbRouteFamily::ACTIVATION;r.arguments={0,0,1,0,3,0,p,2048,8};break;
        case Pi05FullSite::QLinear:
            r.family=FmbRouteFamily::LINEAR;r.selector=19;r.arguments={p,2048,2048,1,8,1,0,3,0,1,b};break;
        case Pi05FullSite::QRope:
            r.family=FmbRouteFamily::ROPE;r.arguments={1,b,p,1,256,256,0,8};break;
        case Pi05FullSite::KvLinear:
            r.family=FmbRouteFamily::LINEAR;r.arguments={0,half*b,1,b+2*d,1,b+2*d+c*64,c,32,2048,8};break;
        case Pi05FullSite::KvCache: {
            r.family=FmbRouteFamily::KV_INSERT;
            const auto plan=rpu_resolve_kvinsert_segment_plan(half*c,c,c,8,8,256,
                KV_INSERT_CAP_V16,KvInsertRoute::ALIGNED_V16);
            r.selector=static_cast<int64_t>(plan.route());
            r.flags=PI05_PREFILL_KV1_DIRECT_ROUTE_FLAG;
            const auto arguments=rpu_kvinsert_route_arguments(plan,8,8,256);
            r.arguments.assign(arguments.begin(),arguments.end());break;
        }
        case Pi05FullSite::Mask:r.arguments={0,0,p,p,4,8};break;
        case Pi05FullSite::Attention:
            r.family=FmbRouteFamily::ATTENTION;r.selector=static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV);
            r.flags=GEMMA_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED;r.arguments={1,p,p,8,8,256,8,4};break;
        case Pi05FullSite::OLinear:
            r.family=FmbRouteFamily::LINEAR;r.arguments={0,pi05_full_mask_bytes(),0,pi05_full_mask_bytes()+d,1,0,1,b,c,2048,256,8};break;
        case Pi05FullSite::OwnerNorm:
            r.family=FmbRouteFamily::ALL_REDUCE;r.arguments={1,half*b,2,half*d,3,half*d,0,half*c*2080,0,p*2080+half*c*2,c,2048,8};break;
        case Pi05FullSite::GateUp:
            r.family=FmbRouteFamily::LINEAR;r.arguments={0,0,0,p*2080,0,pi05_a8_root_bytes(),1,0,1,b,c,p,2048,16384,8};break;
        case Pi05FullSite::DownQuant:
            r.family=FmbRouteFamily::ACTIVATION;r.arguments={1,0,0,0,0,b,p,2048,8};break;
        case Pi05FullSite::DownLinear:
            r.family=FmbRouteFamily::LINEAR;r.selector=27;r.arguments={p,2048,16384,0,8,0,0,0,b,1,0};break;
        case Pi05FullSite::Residual:
            r.family=FmbRouteFamily::ALL_REDUCE;r.arguments={1,half*b,3,half*d,0,half*b,c,2048,8};break;
        case Pi05FullSite::Tail:r.selector=half+1;r.arguments={17,p,1,1,0,0,0};break;
        case Pi05FullSite::Output:r.arguments={0,0,p,2048,1};break;
    }
    return r;
}
std::vector<FmbRouteManifestEntry> pi05_full_routes() const {
    std::vector<FmbRouteManifestEntry> routes;
    for(auto site:{Pi05FullSite::Policy,Pi05FullSite::InputNorm,Pi05FullSite::QQuant,
        Pi05FullSite::QLinear,Pi05FullSite::QRope,Pi05FullSite::Mask,Pi05FullSite::Attention,
        Pi05FullSite::OLinear,Pi05FullSite::GateUp,Pi05FullSite::DownQuant,Pi05FullSite::DownLinear,
        Pi05FullSite::Output}) routes.push_back(pi05_full_route(site));
    for(auto site:{Pi05FullSite::InputCompact,Pi05FullSite::KvLinear,Pi05FullSite::KvCache,
        Pi05FullSite::OwnerNorm,Pi05FullSite::Residual,Pi05FullSite::Tail})
        for(int h=0;h<2;++h) routes.push_back(pi05_full_route(site,h));
    // Only layer zero emits this actual mutable full-P ingress DMA. Later
    // layers read the complete contiguous result from R0 without DDR traffic.
    routes.push_back({FMB_SHARED_LAYER_INPUT_DMA_SITE,FmbRouteFamily::MUTABLE_DMA,
        static_cast<int64_t>(FmbSharedMutableDmaRouteSelector::DDR_BROADCAST_TO_SPM),0,{},0});
    return routes;
}
void pi05_full_consume(Pi05FullSite site,int half=0) const {
    const auto r=pi05_full_route(site,half);
    ctx().consume_physical_route(r.family,r.site_id,r.selector,r.flags,r.arguments,r.invocation);
}
FmbPhysicalExecutionManifest pi05_full_chunk_manifest(const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,int64_t physical_len,int64_t logical_len,int64_t position) const {
    TORCH_CHECK(pi05_full_chunk_domain(plan,layout,physical_len,logical_len,position),
        "RPU_PLANNER_REJECT:CAPABILITY: Pi full chunk requires exact single P544/P800 DDR_KV");
    FmbPhysicalExecutionManifest m;
    m.state=FmbPhysicalManifestState::COMPLETE;m.logical_length=logical_len;m.physical_length=physical_len;
    m.execution_padding_rows=0;m.kv_logical_length=logical_len;m.kv_insert_physical_rows=physical_len;
    m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;m.linear_accumulation=pi05_prefill_linear_accumulation();
    m.routes=pi05_full_routes();
    // The common finalizer canonicalizes (family, site, invocation). The
    // actual shared ingress is already present above, so add no second copy.
    append_fmb_shared_runtime_routes(m,plan,hidden_size(),0);
    return m;
}
void pi05_full_chunk_validate_manifest(const FmbPhysicalExecutionManifest& m) const {
    TORCH_CHECK(pi05_full_chunk_owner() && pi05_kernel_opt_in("RPU_PI05_PREFILL_FULL_CHUNK_W8A8") &&
        m.state==FmbPhysicalManifestState::COMPLETE && m.physical_length==pi05_prefix_rows() &&
        m.logical_length==m.physical_length && m.execution_padding_rows==0 &&
        m.kv_logical_length==m.physical_length && m.kv_insert_physical_rows==m.physical_length &&
        m.graph_lifecycle==FmbGraphLifecycle::COMPOSITE_CHILD && m.linear_accumulation==pi05_prefill_linear_accumulation(),
        "Pi full chunk COMPLETE owner or geometry drift");
    const auto expected=pi05_full_routes();
    TORCH_CHECK(m.routes.size()==expected.size(),"Pi full chunk foreign/missing route");
    for(const auto& e:expected) {
        size_t count=0;
        for(const auto& r:m.routes) count+=r.site_id==e.site_id && r.family==e.family &&
            r.selector==e.selector && r.flags==e.flags && r.arguments==e.arguments && r.invocation==e.invocation;
        TORCH_CHECK(count==1,"Pi full chunk route missing, duplicated or mismatched: ",e.site_id);
    }
}
std::vector<BufferDecl> pi05_full_chunk_decls(const LayoutContext& layout,std::vector<BufferDecl> original) const {
    const bool exact=pi05_full_chunk_owner() && layout.chunk_size==pi05_prefix_rows() &&
        layout.effective_kv_cs()==pi05_prefix_rows() && layout.max_kv_seq_len==pi05_prefix_rows() &&
        layout.batch_size==1 && layout.num_layers==18 && layout.use_attn_mask && !layout.is_causal &&
        layout.attention_policy==AttentionExecutionPolicy::DDR_KV;
    if(!exact) {
        // FMB enumerates C16..P before asking physical_manifest_domain, and
        // persistent reservation also uses an unbound C16 declaration seed.
        // Those are capacity probes, not authority to execute this route.
        // Keep the ordinary declarations for them; the exact physical domain
        // rejects every non-P candidate. Once COMPLETE is bound, fail closed.
        TORCH_CHECK(layout.physical_manifest_fingerprint==0,
            "Pi full chunk bound layout is outside its exact declared geometry: expected_P=",pi05_prefix_rows(),
            " chunk=",layout.chunk_size," kv_chunk=",layout.effective_kv_cs(),
            " max_kv=",layout.max_kv_seq_len," batch=",layout.batch_size,
            " layers=",layout.num_layers," mask=",layout.use_attn_mask," causal=",layout.is_causal,
            " attention=",static_cast<int64_t>(layout.attention_policy),
            " manifest=",layout.physical_manifest_fingerprint," owner=",pi05_full_chunk_owner());
        return original;
    }
    std::vector<BufferDecl> out;
    const auto root=[&](const char* name,int64_t bytes) {
        out.push_back({name,bytes,0,30,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
    };
    root("residual1",pi05_full_root_bytes());root("pi05_full_r1",pi05_full_root_bytes());
    root("pi05_full_rc0",2*pi05_full_d());root("pi05_full_rc1",2*pi05_full_d());
    for(auto& d:original) if(d.storage!=StorageClass::Temp && d.storage!=StorageClass::TempPerLayer)
        out.push_back(std::move(d));
    return out;
}
uint32_t pi05_full_address(int root,int64_t offset,int64_t bytes) {
    TORCH_CHECK(root>=0 && root<4,"Pi full chunk invalid root");
    const char* names[]={"residual1","pi05_full_r1","pi05_full_rc0","pi05_full_rc1"};
    const int64_t extent=root<2?pi05_full_root_bytes():2*pi05_full_d();
    const uint64_t base=addr(0,names[root]),spm_base=SPM_ALLOC.addr(0,0);
    TORCH_CHECK(offset>=0 && bytes>0 && offset<=extent-bytes && base%256==0 &&
        base>=spm_base && base-spm_base+extent<=SpmAllocator::SPM_USABLE &&
        base+offset+bytes<=uint64_t(UINT32_MAX)+1,"Pi full chunk operand escaped declared root");
    return static_cast<uint32_t>(base+offset);
}
SpmOffset pi05_full_offset(int root,int64_t offset,int64_t bytes) {
    const auto absolute=pi05_full_address(root,offset,bytes);
    return SpmOffset{static_cast<uint32_t>(absolute-SPM_ALLOC.addr(0,0))};
}
void pi05_full_require_runtime(int layer,const ChunkInfo& chunk) const {
    TORCH_CHECK(pi05_full_chunk_owner() && layer>=0 && layer<18 && chunk.idx==0 &&
        chunk.offset==0 && chunk.len==pi05_prefix_rows() && chunk.kv_seq_len==pi05_prefix_rows() &&
        ctx().position==0 && ctx().seq_len==pi05_prefix_rows() && ctx().batch_size==1 &&
        ctx().attention_policy==AttentionExecutionPolicy::DDR_KV && prefill_rope_ready_,
        "Pi full chunk runtime owner/plan drift");
    pi05_down_pair_require_graph();
}
void emit_pi05_full_chunk_kv(int layer,const ChunkInfo& chunk) {
    pi05_full_require_runtime(layer,chunk);
    const int64_t c=pi05_pair_rows(),p=pi05_prefix_rows(),b=pi05_full_b(),d=pi05_full_d();
    const auto& w=layer_weights_[layer];
    if(layer==0) pi05_full_consume(Pi05FullSite::Policy);
    if(!ctx().input_in_spm) emit_layer_input_dma(layer,chunk);
    const auto raw=pi05_full_address(0,0,2*b);
    if(layer!=17) for(int half=0;half<2;++half) {
        pi05_full_consume(Pi05FullSite::InputCompact,half);
        rpu_launch_spm_local_shard_copy_dma(raw+half*b,d/2,d,
            pi05_full_address(2,half*d,d),8);
    }
    pi05_full_consume(Pi05FullSite::InputNorm);
    rpu_launch_rmsnorm_spm_kernel(raw,raw,layer_addr(layer,0,"input_norm_w"),p,2048,eps_,
        rmsnorm_spm_contract_.resolve(p,2048),8);
    if(layer!=17) {
        pi05_full_consume(Pi05FullSite::QQuant);
        rpu_launch_dynamic_quantize_w8a8_spm_kernel(raw,pi05_full_address(1,0,b),
            pi05_full_address(3,0,p*2),p,2048,8);
        pi05_full_consume(Pi05FullSite::QLinear);
        rpu_launch_linear_w8a8_spm_kernel(pi05_full_address(1,0,b),pi05_full_address(3,0,p*2),
            w.q_w,w.q_ws,pi05_full_address(1,b,2*d),p,2048,2048,1,8);
        pi05_full_consume(Pi05FullSite::QRope);
        rpu_launch_partial_mrope_spm_kernel(pi05_full_address(1,b,2*d),pi05_full_address(1,b,2*d),
            prefill_cos_,prefill_sin_,0,p,1,256,256,8);
    } else pi05_full_consume(Pi05FullSite::Tail,0);
    for(int half=0;half<2;++half) {
        const int64_t ko=b+2*d,vo=ko+c*64;
        pi05_full_consume(Pi05FullSite::KvLinear,half);
        rpu_launch_pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048_kernel(
            raw+half*b,w.k_w,w.k_ws,pi05_full_address(1,ko,c*64),
            w.v_w,w.v_ws,pi05_full_address(1,vo,c*64),c);
        const auto route=pi05_full_route(Pi05FullSite::KvCache,half);
        const auto plan=rpu_kvinsert_segment_plan_from_route_arguments(route.arguments,8,8,256);
        pi05_full_consume(Pi05FullSite::KvCache,half);
        rpu_launch_pi05_prefill_kv1_direct_cache_m400d256p400_kernel(
            pi05_full_offset(1,ko,c*64),pi05_full_offset(1,vo,c*64),
            (*ctx().k_caches)[layer],(*ctx().v_caches)[layer],prefill_cos_,prefill_sin_,half*c,plan);
    }
}
void emit_pi05_full_chunk_compute(int layer,const ChunkInfo& chunk) {
    pi05_full_require_runtime(layer,chunk);
    if(layer==17) { pi05_full_consume(Pi05FullSite::Tail,1);return; }
    const int64_t c=pi05_pair_rows(),p=pi05_prefix_rows(),b=pi05_full_b(),d=pi05_full_d();
    const auto& w=layer_weights_[layer];
    const int64_t mask=pi05_full_mask_bytes(),tmp=mask+2*d;
    TORCH_CHECK(chunk_masks_.size()==1 && chunk_masks_[0].mask_type==4,
        "Pi full chunk requires exactly one full-prefix explicit mask");
    pi05_full_consume(Pi05FullSite::Mask);
    gemma_upload_chunk_pm(chunk_masks_[0],pi05_full_offset(0,0,mask).value,p,p,8);
    pi05_full_consume(Pi05FullSite::Attention);
    rpu_launch_sdpa_spm_unified_kernel_v2((*ctx().k_caches)[layer],(*ctx().v_caches)[layer],4,c10::nullopt,
        pi05_full_offset(1,b,2*d).value,pi05_full_offset(0,mask,2*d).value,
        pi05_full_offset(0,tmp,pi05_full_tmp_bytes()).value,pi05_full_offset(0,0,mask).value,
        p,8,8,256,p,8,8);
    pi05_full_consume(Pi05FullSite::OLinear);
    rpu_launch_pi05_prefill_pair_spm_kernel(Pi05PrefillPairedProjection::AttentionOutputRow,
        pi05_full_address(0,mask,d),pi05_full_address(0,mask+d,d),w.o_w,
        pi05_full_address(1,0,b),pi05_full_address(1,b,b),w.o_ws,c);
    for(int half=0;half<2;++half) {
        pi05_full_consume(Pi05FullSite::OwnerNorm,half);
        rpu_launch_pi05_owner_norm_a8_spm_kernel(pi05_full_address(1,half*b,b),
            pi05_full_address(2,half*d,d),pi05_full_address(3,half*d,d),
            pi05_full_address(0,half*c*2080,c*2080),layer_addr(layer,0,"post_attn_norm_w"),
            pi05_full_address(0,p*2080+half*c*2,c*2+(c==272?32:0)),eps_,c);
    }
    TORCH_CHECK(pi05_a8_scales_.size()==layer_weights_.size(),"Pi full chunk missing cold GateUp scales");
    const auto& scales=pi05_a8_scales_.at(layer);
    pi05_full_consume(Pi05FullSite::GateUp);
    rpu_launch_pi05_prefill_gate_up_geglu_w8a8_split_kernel(pi05_full_address(0,0,p*2080),
        pi05_full_address(0,p*2080,p*2+30),
        pi05_full_address(0,pi05_a8_root_bytes(),pi05_full_gate_scratch()),w.gate_proj_w,w.up_proj_w,
        pi05_full_address(1,0,b),pi05_full_address(1,b,b),scales[0],scales[1],c,pi05_prefill_a8_panel512_);
    pi05_full_consume(Pi05FullSite::DownQuant);
    rpu_launch_dynamic_quantize_w8a8_spm_kernel(pi05_full_address(1,0,2*b),
        pi05_full_address(0,0,b),pi05_full_address(0,b,p*2),p,2048,8);
    pi05_full_consume(Pi05FullSite::DownLinear);
    rpu_launch_linear_w8a8_spm_kernel(pi05_full_address(0,0,b),pi05_full_address(0,b,p*2),
        w.down_proj_w,w.down_ws,pi05_full_address(1,0,2*b),p,2048,16384,0,8);
    for(int half=0;half<2;++half) {
        pi05_full_consume(Pi05FullSite::Residual,half);
        rpu_launch_pi05_compact_residual_spm_kernel(pi05_full_address(1,half*b,b),
            pi05_full_address(3,half*d,d),pi05_full_address(0,half*b,b),c);
    }
    pi05_full_consume(Pi05FullSite::Output);
    // Full raw output occupies the same R0 as the next single-P input.
    // The terminal layer is cache-only and deliberately has no hidden output.
    if(!ctx().output_to_spm) emit_layer_output_dma(layer,chunk);
}
