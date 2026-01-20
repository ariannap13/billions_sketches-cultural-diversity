from lxml import etree

top_countries = 100

def convert_viz_position_to_attvalues(input_path, output_path):
    tree = etree.parse(input_path)
    root = tree.getroot()

    # Get namespaces
    nsmap = root.nsmap
    viz_ns = nsmap.get('viz', 'http://www.gexf.net/1.2draft/viz')
    gexf_ns = nsmap.get(None)
    ns = {'g': gexf_ns, 'viz': viz_ns}

    # 1. Add attribute definitions to <attributes> section
    attr_section = root.find(".//g:attributes[@class='node']", namespaces=ns)
    if attr_section is None:
        graph_elem = root.find(".//g:graph", namespaces=ns)
        attr_section = etree.SubElement(graph_elem, f"{{{gexf_ns}}}attributes", {"class": "node", "mode": "static"})

    def ensure_attribute(attr_id, title, attr_type="double"):
        existing = attr_section.xpath(f"./g:attribute[@id='{attr_id}']", namespaces=ns)
        if not existing:
            etree.SubElement(attr_section, f"{{{gexf_ns}}}attribute", {"id": attr_id, "title": title, "type": attr_type})

    ensure_attribute("x", "x", "double")
    ensure_attribute("y", "y", "double")

    # 2. Process each node
    for node in root.xpath(".//g:node", namespaces=ns):
        position = node.find(f"{{{viz_ns}}}position")
        if position is not None:
            x = position.get("x")
            y = position.get("y")

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

    # 3. Write the updated GEXF file
    tree.write(output_path, encoding="utf-8", pretty_print=True, xml_declaration=True)
    print(f"✅ Updated GEXF with x, y attributes saved to: {output_path}")

# Example usage
convert_viz_position_to_attvalues(f"../data/image_coordinates_top{top_countries}c.gexf", f"../data/image_coordinates_top{top_countries}c_xy.gexf")
