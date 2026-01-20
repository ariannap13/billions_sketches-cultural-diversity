from lxml import etree

top_countries = 100

def replace_viz_positions_with_attvalues(target_path, reference_path, output_path):
    # Parse both GEXF files
    target_tree = etree.parse(target_path)
    target_root = target_tree.getroot()

    reference_tree = etree.parse(reference_path)
    reference_root = reference_tree.getroot()

    ref_nsmap = reference_root.nsmap
    ref_gexf_ns = ref_nsmap.get(None)
    ref_viz_ns = ref_nsmap.get('viz', 'http://www.gexf.net/1.2draft/viz')
    ref_ns = {'g': ref_gexf_ns, 'viz': ref_viz_ns}
    
    # Extract namespaces
    nsmap = target_root.nsmap
    viz_ns = nsmap.get('viz', 'http://www.gexf.net/1.2draft/viz')
    gexf_ns = nsmap.get(None)
    ns = {'g': gexf_ns, 'viz': viz_ns}

    # Step 1: Collect x and y from reference file using node ID
    ref_positions = {}
    for node in reference_root.xpath(".//g:node", namespaces=ref_ns):
        node_id = node.get("id")
        attvalues = node.find(f"{{{ref_gexf_ns}}}attvalues")
        if attvalues is not None:
            x = y = None
            for attvalue in attvalues.findall(f"{{{ref_gexf_ns}}}attvalue"):
                if attvalue.get("for") == "x":
                    x = attvalue.get("value")
                elif attvalue.get("for") == "y":
                    y = attvalue.get("value")
            if x is not None and y is not None:
                ref_positions[node_id] = (x, y)


    # Step 2: Ensure attributes x and y are defined in <attributes>
    attr_section = target_root.find(".//g:attributes[@class='node']", namespaces=ns)
    if attr_section is None:
        graph_elem = target_root.find(".//g:graph", namespaces=ns)
        attr_section = etree.SubElement(graph_elem, f"{{{gexf_ns}}}attributes", {"class": "node", "mode": "static"})

    def ensure_attribute(attr_id, title, attr_type="double"):
        existing = attr_section.xpath(f"./g:attribute[@id='{attr_id}']", namespaces=ns)
        if not existing:
            etree.SubElement(attr_section, f"{{{gexf_ns}}}attribute", {"id": attr_id, "title": title, "type": attr_type})

    ensure_attribute("x", "x", "double")
    ensure_attribute("y", "y", "double")

    # Step 3: For each node in target file
    for node in target_root.xpath(".//g:node", namespaces=ns):
        node_id = node.get("id")

        # If there is a <viz:position>, move x and y into attvalues (for completeness)
        pos_elem = node.find(f"{{{viz_ns}}}position")
        if pos_elem is not None:
            x = pos_elem.get("x")
            y = pos_elem.get("y")

            attvalues = node.find(f"{{{gexf_ns}}}attvalues")
            if attvalues is None:
                attvalues = etree.SubElement(node, f"{{{gexf_ns}}}attvalues")

            def set_attvalue(attr_id, val):
                existing = attvalues.xpath(f"./g:attvalue[@for='{attr_id}']", namespaces=ns)
                if existing:
                    existing[0].set("value", val)
                else:
                    etree.SubElement(attvalues, f"{{{gexf_ns}}}attvalue", {"for": attr_id, "value": val})

            set_attvalue("x", x)
            set_attvalue("y", y)

        if node_id in ref_positions:
            x_ref, y_ref = ref_positions[node_id]

            attvalues = node.find(f"{{{gexf_ns}}}attvalues")
            if attvalues is None:
                attvalues = etree.SubElement(node, f"{{{gexf_ns}}}attvalues")

            def set_attvalue(attr_id, val):
                existing = attvalues.xpath(f"./g:attvalue[@for='{attr_id}']", namespaces=ns)
                if existing:
                    existing[0].set("value", val)
                else:
                    etree.SubElement(attvalues, f"{{{gexf_ns}}}attvalue", {"for": attr_id, "value": val})

            set_attvalue("x", x_ref)
            set_attvalue("y", y_ref)

    # Step 4: Write the updated file
    target_tree.write(output_path, encoding="utf-8", pretty_print=True, xml_declaration=True)
    print(f"✅ Saved updated file with x, y attvalues replaced: {output_path}")

# Example usage
replace_viz_positions_with_attvalues(
    target_path=f"../data/language_coordinates_top{top_countries}c.gexf",
    reference_path=f"../data/image_coordinates_top{top_countries}c_xy.gexf",
    output_path=f"../data/language_coordinates_top{top_countries}c_xy.gexf"
)
