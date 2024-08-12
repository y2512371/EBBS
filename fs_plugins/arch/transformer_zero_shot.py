from fairseq.models import register_model_architecture

@register_model_architecture("transformer", "transformer_iwslt_zero_shot")
def transformer_iwslt_zero_shot(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    transformer_wmt_en_de(args)


@register_model_architecture("transformer_mod", "transformer_mod_iwslt_zero_shot")
def transformer_iwslt_zero_shot(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    transformer_wmt_en_de(args)
